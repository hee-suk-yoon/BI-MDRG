"""
Preprocess and load datasets for training.
"""

import functools
import io
import json
import math
import re
import random
import numpy as np
import torch
import torchvision
import webdataset as wds
from PIL import Image
import base64
from scipy.optimize import linear_sum_assignment

from data_utils import *
import ipdb 

Image.MAX_IMAGE_PIXELS = 1000000000
N_CHANNELS = 3
MIN_KB = 10
_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


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


def filter_no_caption_or_no_image(sample):
    """
    Filter out LAION samples with no caption or no image.
    """
    return ("txt" in sample) and (
        "png" in sample or "jpg" in sample or "jpeg" in sample
    )


def preprocess_laion_text(sample, tokenizer, max_tokens=32):
    """
    Preprocess text for LAION.
    Captions are truncated to 32 tokens by default.
    """
    tokenizer.padding_side = "right"
    sample = [
        (f"<image>{s.strip()}<|endofchunk|>{tokenizer.eos_token}") for s in sample
    ]
    text = tokenizer(
        sample,
        max_length=max_tokens,
        padding="longest",
        truncation="only_first",
        return_tensors="pt",
    )
    return text["input_ids"], text["attention_mask"]

def preprocess_mmdialogue_text(sample, tokenizer, max_tokens=256):
    """
    Preprocess text for LAION.
    Captions are truncated to 32 tokens by default.
    """
    tokenizer.padding_side = "right"
    sample = [s.strip() for s in sample]
    text = tokenizer(
        sample,
        max_length=max_tokens,
        padding="longest",
        truncation="only_first",
        return_tensors="pt",
    )

    # list of number of image tokens in each sample 
    image_token_id = tokenizer.additional_special_tokens_ids[
        tokenizer.additional_special_tokens.index("<image>")
    ]
    num_image_tokens = torch.count_nonzero(text["input_ids"] == image_token_id, dim=1)

    return text["input_ids"], text["attention_mask"], num_image_tokens


def get_mmdialogue(args, image_processor, tokenizer, epoch=0, floor=False, split='test'): # added by HSY 10/3/23
    import ipdb
    import pickle
    import os
    from torch.utils.data import DataLoader, Dataset
    import cv2
    from torchvision.transforms import functional as TF
    import numpy as np
    from dataclasses import dataclass
    from torch.utils.data.distributed import DistributedSampler


    # open pickle dataset -----
    def open_pickle(data_path, split):
	#open data pickle file at data_path
        with open(os.path.join(data_path,split + '.pkl'), 'rb') as f:
            data = pickle.load(f)
        return data

    # get data
    data = open_pickle(args.data_path, split)
    num_samples = len(data['text'])

    # create a shared epoch store to sync epoch to dataloader worker proc
    shared_epoch = SharedEpoch(epoch=epoch)

    # create two preprocess functions that take in the passed in image_processor and tokenizer
    preprocess_image_fn = functools.partial(preprocess_image, image_processor=image_processor)
    preprocess_text_fn = functools.partial(preprocess_mmdialogue_text, tokenizer=tokenizer)

    class MMDataset(Dataset):
        def __init__(self, data, split):
            self.data = data
            self.split = split

            # Precompute image paths
            self.image_paths = []
            for image_ids in self.data['image_id']:
                paths = [os.path.join(args.image_path, self.split, image_id) for image_id in image_ids]
                self.image_paths.append(paths)

        def __len__(self):
           return len(self.data['text'])

        
        def __getitem__(self, idx):
            # Load images using OpenCV
            images = [cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB) for path in self.image_paths[idx]]
            # Convert to PIL Images for torchvision compatibility
            images = [TF.to_pil_image(img) for img in images]
            text = self.data['text'][idx]
            return (text, images)
        

    @dataclass
    class DataCollator_custom:
        def __init__(self, args, preprocess_image_fn, preprocess_text_fn, tokenizer, split, mask_image_description): #edited by HSY 10/21/23
            #self.data = data
            self.preprocess_image_fn = preprocess_image_fn
            self.preprocess_text_fn = preprocess_text_fn
            self.media_token_id = tokenizer.encode("<image>")[-1]
            self.media_end_token_id = tokenizer.encode("</image>")[-1]
            #self.eos_token_id = tokenizer.encode("<|endofchunk|>")[-1]
            self.eos_token_id = tokenizer.encode("<PAD>")[-1]
            self.split = split

        # make mask_map for langauge model 
        #  image_map and image_feature_length is for making the mask for the cross attention for gated attention layer.
        def create_mask(self,input_ids, media_token_id, media_end_token_id, eos_token_id, image_map=None, image_feature_length=64): 
            # Create a mask filled with True values
            mask = torch.ones(input_ids.size(0), input_ids.size(1), input_ids.size(1), dtype=torch.bool)
            max_images = max(image_map)
            mask_image = torch.zeros(input_ids.size(0), input_ids.size(1), image_feature_length*max_images, dtype=torch.bool)


            # Iterate over each sequence in the batch
            for i, sequence in enumerate(input_ids):
                mask_temp = torch.ones(input_ids.size(1), dtype=torch.bool)
                mask_image_temp = torch.zeros(input_ids.size(1), image_feature_length*max_images, dtype=torch.bool)
                # Find the positions of the media tokens
                start_positions = (sequence == media_token_id).nonzero(as_tuple=True)[0].tolist()
                end_positions = (sequence == media_end_token_id).nonzero(as_tuple=True)[0].tolist()
                eos_start_positions = (sequence == eos_token_id).nonzero(as_tuple=True)[0].tolist()
                if eos_start_positions:
                    eos_start_positions = eos_start_positions[0]
                else:
                    eos_start_positions = None

                # image mask creation----
                for ii, end_position in enumerate(end_positions):
                    if eos_start_positions:
                        mask_image_temp[end_position+1:eos_start_positions,image_feature_length*(ii):image_feature_length*(ii+1)] = True
                    else:
                        mask_image_temp[end_position+1:,image_feature_length*(ii):image_feature_length*(ii+1)] = True
                #------------------------

                start_positions_temp = []
                end_positions_temp = []
                while start_positions:
                    start = start_positions.pop(0)
                    start_positions_temp.append(start+1)
                    # If no corresponding end token is found, mask till the end
                    if not end_positions:
                        if eos_start_positions:
                            mask_temp[start+1:eos_start_positions] = False
                            end_positions_temp.append(eos_start_positions-1)
                        else:
                            mask_temp[start+1:] = False
                            end_positions_temp.append(input_ids.size(1)-1)
                        break
                    else:
                        end = end_positions.pop(0)
                        end_positions_temp.append(end-1)
                        mask_temp[start+1:end] = False                
                mask_temp = mask_temp.unsqueeze(0).repeat(mask_temp.size(0),1)  
                for j in range(len(start_positions_temp)):
                    mask_temp[start_positions_temp[j]:end_positions_temp[j]+1, :] = True

                mask[i, :, :] = mask_temp
                mask_image[i, :, :] = mask_image_temp
            
            mask_image = mask_image.unsqueeze(1)
            #mask is for normal layers in the language model, and mask_image is for the gated layer in the language model

            # for batch with no image.
            if max_images == 0:
                mask_image = torch.zeros(input_ids.size(0), 1, input_ids.size(1), image_feature_length, dtype=torch.bool)

            return mask, mask_image

        # this right now has bug. fix before use
        def create_mask_parallel(self,input_ids, media_token_id, media_end_token_id, eos_token_id):
            # Create a mask filled with True values
            mask = torch.ones_like(input_ids, dtype=torch.bool)
            
            # Find the positions of media tokens and eos tokens
            start_positions = (input_ids == media_token_id).nonzero(as_tuple=True)
            end_positions = (input_ids == media_end_token_id).nonzero(as_tuple=True)
            eos_positions = (input_ids == eos_token_id).nonzero(as_tuple=True)
            
            # Initialize markers for all sequences in the batch
            last_start = -1 * torch.ones((input_ids.size(0),), dtype=torch.long)
            last_end = -1 * torch.ones((input_ids.size(0),), dtype=torch.long)
            last_eos = input_ids.size(1) * torch.ones((input_ids.size(0),), dtype=torch.long)
            
            # Update markers with the positions of the tokens
            if eos_positions[0].numel() > 0:
                last_eos[eos_positions[0]] = eos_positions[1]
            
            for batch_idx, seq_idx in zip(*start_positions):
                last_start[batch_idx] = max(last_start[batch_idx], seq_idx)
            
            for batch_idx, seq_idx in zip(*end_positions):
                last_end[batch_idx] = max(last_end[batch_idx], seq_idx)
            
            # Set mask values based on the positions of the tokens
            for i in range(input_ids.size(0)):
                if last_start[i] > last_end[i]:
                    mask[i, last_start[i]:last_eos[i]] = False
                else:
                    mask[i, last_start[i]:last_end[i]+1] = False
                    
            return mask

        def __call__(self, pre_batch):
            texts = [sample[0] for sample in pre_batch]
            texts_processed = self.preprocess_text_fn(texts)


            imgs_all = [sample[1] for sample in pre_batch]
            imgs_pruned = [img for i in range(len(imgs_all)) for img in imgs_all[i][:texts_processed[-1].tolist()[i]]] #pruned based on truncated text where the <image> token is gone            
            try:
                imgs_processed = self.preprocess_image_fn(imgs_pruned)
            except:
                imgs_processed = torch.empty((0, 3, 224, 224)) 
            mask_map, mask_image_map = self.create_mask(texts_processed[0], self.media_token_id, self.media_end_token_id, self.eos_token_id, texts_processed[2].tolist(), 64)
            return texts_processed, imgs_processed, mask_map, mask_image_map

    # Configure DataLoader  
    data = MMDataset(data, split=split)
    sampler = DistributedSampler(data, num_replicas=args.world_size, rank=args.rank, shuffle=False, drop_last=True)
    dataloader = DataLoader(data, batch_size=args.batch_size_mmdialogue, shuffle=False, sampler=sampler, num_workers=args.workers, collate_fn= DataCollator_custom(args, preprocess_image_fn, preprocess_text_fn, split=split, tokenizer=tokenizer, mask_image_description=args.mask_image_description), drop_last=True, persistent_workers=True)
    # add meta-data to dataloader instance for convenience
    dataloader.num_samples = num_samples
    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)

def get_dataset_fn(dataset_type):
    """
    Helper function to get the dataset function based on the dataset type
    """
    if dataset_type == "mmdialogue":
        return get_mmdialogue
    # elif dataset_type == "mmdialogue_single_gpu":
    #     return get_mmdialogue_single_gpu
    # elif dataset_type == "mmdialogue_single_gpu_split":
    #     return get_mmdialogue_single_gpu_split

    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def get_data(args, image_processor, tokenizer, dataset_type, epoch=0, split=None):
    """
    Interface for getting the webdatasets
    """
    return get_dataset_fn(dataset_type)(
        args, image_processor=image_processor, epoch=epoch, tokenizer=tokenizer, split = split
    )



