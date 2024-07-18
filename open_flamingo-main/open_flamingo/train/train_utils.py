import time
from contextlib import suppress
import torch
from tqdm import tqdm
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.api import FullOptimStateDictConfig
import os
import wandb
from einops import rearrange
import torchvision

def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == "bf16":
        cast_dtype = torch.bfloat16
    elif precision == "fp16":
        cast_dtype = torch.float16
    return cast_dtype


def get_mp_policy_dtype(precision: str):
    if "bfloat16" in precision or "bf16" in precision:
        return torch.bfloat16
    elif precision == "fp16":
        return torch.float16
    else:
        return torch.float32


def get_autocast(precision, cache_enabled=True):
    if precision == "amp":
        return torch.cuda.amp.autocast(cache_enabled=cache_enabled)
    elif precision == "amp_bfloat16" or precision == "amp_bf16":
        # amp_bfloat16 is more stable than amp float16 for clip training
        return lambda: torch.cuda.amp.autocast(
            dtype=torch.bfloat16, cache_enabled=cache_enabled
        )
    else:
        return suppress

def train_one_epoch(
    args,
    model,
    epoch,
    mmdialogue_loader,
    mmdialogue_val_loader,
    tokenizer,
    optimizer,
    lr_scheduler,
    device_id,
    total_training_steps, 
    num_batches_per_epoch, 
    num_batches_per_epoch_val, 
    wandb,
    csv_logger, 
    val_csv_logger 
):
    autocast = get_autocast(
        args.precision, cache_enabled=(not args.fsdp)
    )  # if fsdp, disable cache to save memory
    cast_dtype = get_cast_dtype(args.precision)

    # setup model
    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    media_end_token_id = tokenizer("</image>", add_special_tokens=False)["input_ids"][-1] 
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)["input_ids"][-1]

    if args.citation_module:
        cite_token_id = tokenizer("<cite>", add_special_tokens=False)["input_ids"][-1]
        cite_end_token_id = tokenizer("</cite>", add_special_tokens=False)["input_ids"][-1]

    model.train()

    # setup logging
    step_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    lowest_val = 1000
    # loop through dataloader
    for num_steps, (texts, images, mask_map, mask_image_map) in tqdm(
        enumerate(mmdialogue_loader),
        disable=args.rank != 0,
        total=num_batches_per_epoch,
        initial=0,
    ):
        data_time_m.update(time.time() - end)

        #### MMDialogue FORWARD PASS ####
        images = images.to(device_id, dtype=cast_dtype, non_blocking=True)
        images = rearrange(images, "(b t f) c h w -> b t f c h w", t=1, f=1)

        input_ids = texts[0].to(device_id, dtype=cast_dtype, non_blocking=True)
        attention_mask = texts[1].to(
            device_id, dtype=cast_dtype, non_blocking=True
        )
        image_map = texts[2].tolist()

        # set up labels; language model is expected to handle shifting
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        labels = labels.to(device_id)

        mask_map = mask_map.to(device_id)
        mask_image_map = mask_image_map.to(device_id)

        # gradient accumulation w/ fsdp cpu offloading requires a no_sync context manager
        with autocast():
            loss = model(
                vision_x=images,
                lang_x=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                image_map=image_map,
                mask_map=mask_map,
                mask_image_map=mask_image_map,
            )[0]
        
        loss.backward()


        if (not args.freeze_lm_embeddings) and (
            not args.fsdp or args.fsdp_use_orig_params
        ):
            # Mask gradients for input embeddings s.t. we only update the added tokens <image> and <|endofchunk|>
            if args.fsdp:
                embed_grad = model.lang_encoder.get_input_embeddings().weight.grad
            else:
                embed_grad = (
                    model.module.lang_encoder.get_input_embeddings().weight.grad
                )
            zero_mask = torch.zeros_like(embed_grad)
            zero_mask[media_token_id] = torch.ones_like(zero_mask[media_token_id])
            zero_mask[endofchunk_token_id] = torch.ones_like(
                zero_mask[endofchunk_token_id]
            )

            zero_mask[media_end_token_id] = torch.ones_like(zero_mask[media_end_token_id])

            if args.citation_module:
                zero_mask[cite_token_id] = torch.ones_like(zero_mask[cite_token_id])
                zero_mask[cite_end_token_id] = torch.ones_like(zero_mask[cite_end_token_id])
            
            if args.fsdp:
                model.lang_encoder.get_input_embeddings().weight.grad = (
                    embed_grad * zero_mask
                )
            else:
                model.module.lang_encoder.get_input_embeddings().weight.grad = (
                    embed_grad * zero_mask
                )
        
        # clip gradient norm
        if args.fsdp:
            """
            The way we clip gradients with FSDP is different than the non-FSDP case,
            because during FSDP, gradient norms are computed over certain submodules,
            rather than the entire model.
            At least for OPT-125M, this didn't seem to make a difference in performance.
            """
            model.clip_grad_norm_(1.0)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # step optimizer and log
        if (((num_steps + 1) % args.gradient_accumulation_steps) == 0) or (
            num_steps == num_batches_per_epoch - 1
        ):
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            # step time and reset end outside of rank 0
            step_time_m.update(time.time() - end)
            end = time.time()

            # rank 0 logging
            if args.rank == 0:
                global_step = num_steps + epoch * num_batches_per_epoch
                loss = loss.item()

                # log to csv 
                csv_logger.log(epoch=epoch,step=global_step,loss=loss)

        if ((num_steps + 1) % 500 == 0):
            model.eval()
            mmdialogue_val_loader.set_epoch((epoch*num_batches_per_epoch)+num_steps)
            mmdialogue_val_loader_ = mmdialogue_val_loader.dataloader  
            with torch.no_grad():
                val_loss = valid_one_epoch_with_loss_return(
                    args=args,
                    model=model,
                    epoch=epoch,
                    step_=(epoch*num_batches_per_epoch)+num_steps,
                    tokenizer=tokenizer,
                    mmdialogue_loader=mmdialogue_val_loader_,
                    device_id=device_id,
                    num_batches_per_epoch=num_batches_per_epoch_val,
                    csv_logger=val_csv_logger
                )
            if val_loss <= lowest_val:
                lowest_val = val_loss
                save_checkpoint_for_resume(model, optimizer, lr_scheduler, epoch, num_steps, args)
            model.train()
       

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def filter_state_dict_to_trainable(model, state_dict):
    """
    Remove non-trainable parameters from model state dict.
    Exception: Embeddings will not be removed, even if frozen.
    This is because we need the new <image> <|endofchunk|> tokens to
    be consistent across initializations.
    """
    for (
        name,
        p,
    ) in model.named_parameters():  # won't work for fsdp + use_orig_params=False
        if "fsdp" in name:
            continue
        if "embed" in name or isinstance(p, torch.nn.Embedding):
            continue
        if not p.requires_grad:
            name = name.replace("._checkpoint_wrapped_module", "")
            if name in state_dict:
                del state_dict[name]
            else:
                print(f"WARNING: filtering but {name} not in state_dict")

    # also remove the keys in state_dict generated from
    # lang_encoder.old_decoder_blocks and lang_encoder.gated_cross_attn_layers
    # because these are already saved in lang_encoder.model...
    to_delete = [
        n
        for n in state_dict.keys()
        if ("lang_encoder.old_decoder_blocks" in n)
        or ("lang_encoder.gated_cross_attn_layers" in n)
        or ("vision_encoder" in n)
    ]
    for name in to_delete:
        del state_dict[name]
    return state_dict

def save_checkpoint_for_resume(model, optimizer, lr_scheduler, epoch, steps, args):
    """
    Save training checkpoint with model, optimizer, and lr_scheduler state.
    """
    if args.fsdp:
        FSDP.set_state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            FullOptimStateDictConfig(rank0_only=True),
        )
        model_state = model.state_dict()
        optim_state = FSDP.optim_state_dict(model, optimizer, group=args.my_group)

    else:
        model_state = model.state_dict()
        optim_state = optimizer.state_dict()

    if args.rank == 0:
        if not (args.fsdp and not args.fsdp_use_orig_params):
            model_state = filter_state_dict_to_trainable(model, model_state)

        if not os.path.exists(args.run_name):
            os.makedirs(args.run_name)

        checkpoint_dict = {
            "epoch": epoch,
            "steps": steps,
            "model_state_dict": model_state,
            "optimizer_state_dict": optim_state,
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        }

        print(f"Saving checkpoint to {args.run_name}/checkpoint_{epoch}.pt")
        torch.save(checkpoint_dict, f"{args.run_name}/checkpoint_{epoch}.pt")

def valid_one_epoch_with_loss_return(
    args,
    model,
    epoch,
    step_,
    mmdialogue_loader,
    tokenizer,
    device_id,
    num_batches_per_epoch,
    csv_logger,
):
    autocast = get_autocast(
        args.precision, cache_enabled=(not args.fsdp)
    )  # if fsdp, disable cache to save memory
    cast_dtype = get_cast_dtype(args.precision)

    # setup model
    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    media_end_token_id = tokenizer("</image>", add_special_tokens=False)["input_ids"][-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)["input_ids"][-1]

    # setup logging
    step_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    loss_total = 0
    # loop through dataloader
    for num_steps, (texts, images, mask_map, mask_image_map) in tqdm(
        enumerate(mmdialogue_loader),
        disable=args.rank != 0,
        total=num_batches_per_epoch,
        initial=0,
    ):
        data_time_m.update(time.time() - end)

        #### MMDialogue FORWARD PASS ####
        images = images.to(device_id, dtype=cast_dtype, non_blocking=True)
        images = rearrange(images, "(b t f) c h w -> b t f c h w", t=1, f=1)

        input_ids = texts[0].to(device_id, dtype=cast_dtype, non_blocking=True)
        attention_mask = texts[1].to(
            device_id, dtype=cast_dtype, non_blocking=True
        )
        image_map = texts[2].tolist()

        # set up labels; language model is expected to handle shifting
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        labels[labels == tokenizer.eos_token] = -100
        labels = labels.to(device_id)

        mask_map = mask_map.to(device_id)
        mask_image_map = mask_image_map.to(device_id)

        # gradient accumulation w/ fsdp cpu offloading requires a no_sync context manager
        with autocast():
            loss = model(
                vision_x=images,
                lang_x=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                image_map=image_map,
                mask_map=mask_map,
                mask_image_map=mask_image_map,
            )[0]
        

        loss_total+=loss.item()

    if args.rank == 0:
        csv_logger.log(epoch=epoch,step=step_,loss=loss_total/num_batches_per_epoch)

    return loss_total/num_batches_per_epoch

def preprocess_image(sample, image_processor):
    """
    Convert images to tensors for training.
    Augmentations: random horizontal flip.
    Normalization handled by wds.
    """
    image = [image_processor(s).unsqueeze(0) for s in sample]
    image = torch.cat(image, dim=0)
    image = torchvision.transforms.RandomHorizontalFlip(p=0.5)(image)
    return image


import logging
def get_logger(args):
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')

    file_handler = logging.FileHandler(os.path.join(args.test_save_path, f"{args.save_run_name}.txt"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger