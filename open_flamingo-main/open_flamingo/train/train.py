""" Main training script """

import argparse
import glob
import os
import random

import numpy as np
import torch
import wandb
from data import get_data
from distributed import init_distributed_device, world_info_from_env
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from train_utils import (
    train_one_epoch,
    get_mp_policy_dtype,
)
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from torch.distributed.fsdp import (
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointWrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
from torch.distributed.fsdp._init_utils import _init_intra_and_inter_node_groups
from torch.distributed.distributed_c10d import _get_default_group
import functools

from custom_files.custom_factory import create_model_and_transforms

import ipdb

def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def main():
    parser = argparse.ArgumentParser()
    # model configuration args
    parser.add_argument("--vision_encoder_path", default="ViT-L-14", type=str)
    parser.add_argument("--vision_encoder_pretrained", default="openai", type=str)
    parser.add_argument("--lm_path", default="facebook/opt-1.3b", type=str)
    parser.add_argument(
        "--tokenizer_path",
        default="facebook/opt-30b",
        type=str,
        help="path to tokenizer",
    )
    parser.add_argument(
        "--cross_attn_every_n_layers",
        type=int,
        default=1,
        help="how often to add a cross-attention layer after each transformer layer",
    )

    # training args
    parser.add_argument(
        "--run_name",
        type=str,
        default="openflamingo3B",
        help="used to name saving directory and wandb run",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        help="path to checkpoint to resume from, this should contain model, optimizer, and lr_scheduler states. if there exists a checkpoint in the dir named run_name, we will resume from that checkpoint by default",
        default=None,
    )
    parser.add_argument(
        "--delete_previous_checkpoint",
        action="store_true",
        help="delete previous checkpoint when saving new checkpoint",
    )

    parser.add_argument("--batch_size_mmdialogue", type=int, default=128)

    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument(
        "--lr_scheduler",
        default="constant",
        type=str,
        help="constant, linear, or cosine",
    )
    parser.add_argument("--loss_multiplier_mmc4", type=float, default=1.0)
    parser.add_argument("--loss_multiplier_laion", type=float, default=1.0)
    parser.add_argument("--warmup_steps", default=5000, type=int)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument(
        "--precision",
        choices=["amp_bf16", "amp_bfloat16", "bf16", "fp16", "fp32"],
        default="fp32",
        help="Floating point precision.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="whether to train with gradient/activation checkpointing",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="we define an 'epoch' as a fixed number of examples (train_num_samples_mmc4, train_num_samples_laion), not a pass through the entire dataset",
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--freeze_lm_embeddings",
        action="store_true",
        help="if True, we freeze the LM embeddings during training. Otherwise, we train the <image> and <|endofchunk|> embeddings.",
    )
    parser.add_argument(
        "--logging_steps", type=int, default=100, help="log loss every n steps"
    )

    # data args
    parser.add_argument(
        "--laion_shards",
        type=str,
        help="path to laion shards, this should be a glob pattern such as /path/to/shards/shard-{0000..0999}.tar",
    )
    parser.add_argument(
        "--mmc4_shards",
        type=str,
        help="path to c4 shards, this should be a glob pattern such as /path/to/shards/shard-{0000..0999}.tar",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--train_num_samples_mmdialogue", type=int, default=10000) 

    parser.add_argument("--dataset_resampled", action="store_true")
    parser.add_argument(
        "--mmc4_textsim_threshold",
        default=30,
        type=float,
        help="threshold for filtering images in mmc4 based on image-text similarity",
    )
    parser.add_argument(
        "--mmc4_max_num_images",
        default=6,
        type=int,
        help="max number of images per sequence in mmc4 / chatgpt",
    )
    parser.add_argument(
        "--mmc4_min_num_images",
        default=1,
        type=int,
        help="min number of images per sequence in mmc4 / chatgpt",
    )

    # distributed training args
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--horovod",
        default=False,
        action="store_true",
        help="Use horovod for distributed training.",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).",
    )
    parser.add_argument(
        "--fsdp",
        default=False,
        action="store_true",
        help="Use FullyShardedDataParallel for distributed training.",
    )
    parser.add_argument(
        "--fsdp_use_orig_params",
        default=False,
        action="store_true",
        help="Passed into the FSDP constructor. Enables param_groups and gradient masking for weight_decay. Does not work with OPT.",
    )
    parser.add_argument(
        "--fsdp_sharding_strategy", default="full", type=str, choices=["full", "hybrid"]
    )

    # wandb args
    parser.add_argument("--report_to_wandb", default=False, action="store_true")
    parser.add_argument(
        "--wandb_project",
        type=str,
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
    )
    parser.add_argument(
        "--save_checkpoints_to_wandb",
        default=False,
        action="store_true",
        help="save checkpoints to wandb",
    )

    #--------------------------------
    parser.add_argument(
        "--image_path",
        type=str,
        default="",
        help="path to images",
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default="",
    )

    parser.add_argument(
        '--citation_module', #whether to use the citation augmented input.
        action='store_true',
    )

    parser.add_argument(
        '--baseline_no_image', #whether to use the visual cross attention module.
        action='store_true',
    )

    parser.add_argument(
        '--full_tuning', 
        action='store_true',
    )
    # ------------------------------

    args = parser.parse_args()

    if args.save_checkpoints_to_wandb and not args.report_to_wandb:
        raise ValueError("save_checkpoints_to_wandb requires report_to_wandb")

    if args.fsdp and not args.fsdp_use_orig_params:
        print(
            "Warning: FSDP is running without fsdp_use_orig_params flag. "
            + "This is not recommended because it means we will use uniform weight decay"
            + " and train all embeddings, not just the newly added ones. "
            + "Note: OPT models are not compatible with fsdp_use_orig_params flag."
        )

    if args.fsdp and args.fsdp_sharding_strategy == "hybrid":
        print(
            "Warning: As of torch=2.0.1, the FSDP logic for optim_state_dict() is broken for hybrid sharding."
            + "To make this method work, we need to modify torch.distributed.fsdp._optim_utils.py"
            + "Copy and paste the code from the _optim_utils.py in this repo into the torch file."
            + "The main issue was the missing group kwarg on line 1596 in _all_gather_optim_state."
        )

    # Set up distributed training
    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    device_id = init_distributed_device(args)
    random_seed(args.seed)

    # Initialize model
    model, image_processor, tokenizer = create_model_and_transforms(
        args,
        args.vision_encoder_path,
        args.vision_encoder_pretrained,
        args.lm_path,
        args.tokenizer_path if args.tokenizer_path else args.lm_path,
        cross_attn_every_n_layers=args.cross_attn_every_n_layers,
        use_local_files=args.offline,
        gradient_checkpointing=args.gradient_checkpointing,
        freeze_lm_embeddings=args.freeze_lm_embeddings,
    )
    random_seed(args.seed, args.rank)

    # Initialize logging
    print(f"Start running training on rank {args.rank}.")
    if args.rank == 0 and args.report_to_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=vars(args),
        )



    # Load model checkpoint on CPU
    if os.path.exists(f"{args.run_name}") and args.resume_from_checkpoint is None:
        # if args do not specify a checkpoint to resume from, check if checkpoints exist for this run
        # and automatically resume from the latest checkpoint
        checkpoint_list = glob.glob(f"{args.run_name}/checkpoint_*.pt")
        if len(checkpoint_list) == 0:
            print(f"Found no checkpoints for run {args.run_name}.")
        else:
            args.resume_from_checkpoint = sorted(
                checkpoint_list, key=lambda x: int(x.split("_")[-1].split(".")[0])
            )[-1]
            print(
                f"Found checkpoint {args.resume_from_checkpoint} for run {args.run_name}."
            )

    resume_from_epoch = 0
    if args.resume_from_checkpoint is not None:
        if args.rank == 0:
            print(f"Loading checkpoint from {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
        try:
            msd = checkpoint["model_state_dict"]
            msd = {k.replace("module.", ""): v for k, v in msd.items()}
            resume_from_epoch = checkpoint["epoch"]
            resume_step = checkpoint["steps"] + 1
        except:
            msd = checkpoint
            resume_step = 0

        # for fsdp, only one rank needs to load the state dict
        if not args.fsdp or args.rank == 0:
            model.load_state_dict(msd, False)
        #----------------------------------
    
    #--------------------------------------------------------
    if args.citation_module:
        pre_additional_special_tokens = tokenizer.additional_special_tokens
        post_additional_special_tokens = pre_additional_special_tokens + ["</image>", "<cite>", "</cite>"]
        tokenizer.add_special_tokens({"additional_special_tokens": post_additional_special_tokens})
        model.lang_encoder.resize_token_embeddings(len(tokenizer))

        model.media_end_token_id = tokenizer.encode("</image>")[-1]
        model.lang_encoder.media_end_token_id = model.media_end_token_id

        model.lang_encoder._encode_special_tokens()
    else:
        pre_additional_special_tokens = tokenizer.additional_special_tokens
        post_additional_special_tokens = pre_additional_special_tokens + ["</image>"]
        tokenizer.add_special_tokens({"additional_special_tokens": post_additional_special_tokens})
        model.lang_encoder.resize_token_embeddings(len(tokenizer))

        model.media_end_token_id = tokenizer.encode("</image>")[-1]
        model.lang_encoder.media_end_token_id = model.media_end_token_id

        model.lang_encoder._encode_special_tokens()
    #--------------------------------------------------------

    # Initialize FSDP / DDP, and ensure the model is on GPU
    print(f"Initializing distributed training with {args.world_size} GPUs.")
    if args.fsdp:
        print(
            f"Before FSDP parameter num: {sum(p.numel() for p in model.parameters())} on rank {args.rank}"
        )

        # init MixedPrecision
        if args.precision != "fp32":
            cast_dtype = get_mp_policy_dtype(args.precision)
            mp_policy = MixedPrecision(
                param_dtype=torch.float32,
                reduce_dtype=cast_dtype,  # gradient communication
                buffer_dtype=cast_dtype,
            )
        else:
            mp_policy = None

        # init process groups
        if args.fsdp_sharding_strategy == "hybrid":
            intra_node_group, inter_node_group = _init_intra_and_inter_node_groups(
                _get_default_group()
            )
            args.my_group = intra_node_group  # for optimizer saving
            process_group = (intra_node_group, inter_node_group)  # for FSDP init
        else:
            args.my_group = None  # for optimizer saving
            process_group = None  # for FSDP init

        # init FSDP
        wrapper_kwargs = dict(
            process_group=process_group,
            cpu_offload=CPUOffload(offload_params=False),
            device_id=device_id,
            sync_module_states=True,  # broadcast loaded ckpt from rank 0 -> all ranks
            sharding_strategy=ShardingStrategy.FULL_SHARD
            if args.fsdp_sharding_strategy == "full"
            else ShardingStrategy.HYBRID_SHARD,
            use_orig_params=args.fsdp_use_orig_params,
            mixed_precision=mp_policy,
            forward_prefetch=True,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            limit_all_gathers=True,
        )
        model.wrap_fsdp(wrapper_kwargs, device_id)
        ddp_model = model

        print(
            f"After FSDP parameter num: {sum(p.numel() for p in model.parameters())} on rank {args.rank}"
        )
        print(
            f"After FSDP {torch.cuda.memory_allocated()/1024**3:.3} GB on rank {args.rank}"
        )

    else:
        model = model.to(device_id)
        ddp_model = DDP(model, device_ids=[device_id])

    # Initialize gradient checkpointing
    if args.gradient_checkpointing:
        non_reentrant_wrapper = functools.partial(
            checkpoint_wrapper,
            offload_to_cpu=True,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        apply_activation_checkpointing(
            ddp_model,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=lambda m: getattr(m, "_use_gradient_checkpointing", False)
            and not isinstance(m, FSDP)
            and not isinstance(m, CheckpointWrapper),
        )

    # Initialize optimizer
    params_to_optimize = ddp_model.named_parameters()
    params_to_optimize = list(
        filter(
            lambda x: x[1].requires_grad
            and not getattr(x[1], "exclude_from_optimizer", False),
            params_to_optimize,
        )
    )
    if not args.fsdp or args.fsdp_use_orig_params:
        # apply weight decay only to params in the xattn layers
        def get_grouped_params(model):
            params_with_wd, params_without_wd = [], []
            for n, p in params_to_optimize:
                if "gated_cross_attn" in n:
                    params_with_wd.append(p)
                else:
                    params_without_wd.append(p)
            return [
                {"params": params_with_wd, "weight_decay": args.weight_decay},
                {"params": params_without_wd, "weight_decay": 0.0},
            ]

        optimizer = torch.optim.AdamW(
            get_grouped_params(params_to_optimize), lr=args.learning_rate
        )
    else:
        # unclear if we should be using no weight decay or small weight decay for all parameters
        optimizer = torch.optim.AdamW(
            (p for _, p in params_to_optimize),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

    # load optimizer checkpoint
    if args.resume_from_checkpoint is not None:
        try:
            osd = checkpoint["optimizer_state_dict"]
            if args.fsdp:
                osd = FSDP.optim_state_dict_to_load(osd, ddp_model, optimizer)
            optimizer.load_state_dict(osd)
        except:
            pass

    # Initialize data loaders
    mmdialogue_dataset = get_data(args, image_processor, tokenizer, "mmdialogue", split='train') 

    mmdialogue_loader_test = get_data(args, image_processor, tokenizer, "mmdialogue", split='test')

    num_batches_per_epoch = len(mmdialogue_dataset.dataloader) 
    total_training_steps = num_batches_per_epoch*(args.num_epochs)

    num_batches_per_epoch_test = len(mmdialogue_loader_test.dataloader)

    if args.rank == 0:
        print(f"Total training steps: {total_training_steps}")

    # Initialize lr scheduler
    if args.lr_scheduler == "linear":
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_training_steps,
        )
    elif args.lr_scheduler == "cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_training_steps,
        )
    else:
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps
        )

    # load lr scheduler checkpoint
    if args.resume_from_checkpoint is not None:
        try:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        except:
            pass

    # Start training!
    ddp_model.train()

    # -------
    if args.rank == 0:
        import csv
        class CSVLogger:
            def __init__(self, filename, fieldnames=['epoch', 'step', 'loss']):
                self.filename = filename
                self.fieldnames = fieldnames
                with open(self.filename, 'w', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                    writer.writeheader()

            def log(self, **kwargs):
                with open(self.filename, 'a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                    writer.writerow(kwargs)
        
        csv_logger = CSVLogger(f"{args.run_name}/loss.csv")
        val_csv_logger = CSVLogger(f"{args.run_name}/val_loss.csv")
    else:
        csv_logger = None
        val_csv_logger = None
    # -------

    from tqdm import tqdm
    for epoch in tqdm(range(resume_from_epoch, args.num_epochs)):
        mmdialogue_dataset.set_epoch(epoch)
        mmdialogue_loader = mmdialogue_dataset.dataloader  

        train_one_epoch(
            args=args,
            model=ddp_model,
            epoch=epoch,
            tokenizer=tokenizer,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            mmdialogue_loader=mmdialogue_loader,
            mmdialogue_val_loader=mmdialogue_loader_test, 
            device_id=device_id,
            total_training_steps=total_training_steps, 
            num_batches_per_epoch=num_batches_per_epoch, 
            num_batches_per_epoch_val=num_batches_per_epoch_test,
            wandb=wandb,
            csv_logger=csv_logger,
            val_csv_logger = val_csv_logger
        )

if __name__ == "__main__":
    main()
