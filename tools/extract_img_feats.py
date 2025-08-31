# ------------------------------------------------------------------------------ #
# Author: Zhenwei Shao (https://github.com/ParadoxZW)
# Description: Tool for extracting image features
# ------------------------------------------------------------------------------ #

import os, sys
sys.path.append(os.getcwd())
import logging
import glob, re, math, time, datetime
import numpy as np
import torch
from torch import nn
from PIL import Image
import clip
from tqdm import tqdm
import argparse
from pathlib import Path
from misc import create_logging, create_output_folders
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf

from math import ceil
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import ImageOps, Image

logger = get_logger(__name__, log_level="INFO")

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC



def Pad():
    def _pad(image):
        W, H = image.size # debugged
        if H < W:
            pad_H = ceil((W - H) / 2)
            pad_W = 0
        else:
            pad_H = 0
            pad_W = ceil((H - W) / 2)
        img = ImageOps.expand(image, border=(pad_W, pad_H, pad_W, pad_H), fill=0)
        # print(img.size)
        return img
    return _pad

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def identity(x):
    return x


def _transform(n_px, pad=False, crop=False):
    return Compose([
        Pad() if pad else identity,
        Resize([n_px, n_px], interpolation=BICUBIC),
        CenterCrop(n_px) if crop else identity,
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

@torch.no_grad()
def _extract_feat(img_path, net, T, save_path):
    # print(img_path)
    img = Image.open(img_path)
    # W, H = img.size
    img = T(img).unsqueeze(0).cuda()
    clip_feats = net(img).cpu().numpy()[0]
    clip_feats = clip_feats.transpose(1, 2, 0) # Dim 
    # print(clip_feats.shape, save_path)
    # return
    import pdb; pdb.set_trace()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        save_path,
        x=clip_feats,
    )


class ExtractModel:
    def __init__(self, encoder) -> None:
        encoder.attnpool = nn.Identity()
        self.backbone = encoder

        self.backbone.cuda().eval()
    
    @torch.no_grad()
    def __call__(self, img):
        x = self.backbone(img)
        return x


def main(
        TASK_TO_IMG_MAPPING: str,
        DATASET: str,
        IMAGE_DIR: dict,
        FEATS_DIR: dict,
        CLIP_NAME: str,
        CLIP_MODEL: str,
        IMG_RESOLUTION: int,
        **kwargs    
         ):
    
    # find imgs
    img_dir_list = []
    feat_dir_list = []
    for split in TASK_TO_IMG_MAPPING:
        if split.startswith(DATASET):
            img_dir_list.append(
                IMAGE_DIR[TASK_TO_IMG_MAPPING[split]]
            )
            feat_dir_list.append(
                FEATS_DIR[TASK_TO_IMG_MAPPING[split]]
            )
    for i, feat_dir in enumerate(feat_dir_list):
        feat_dir = feat_dir.replace('_feats/', f'_{CLIP_NAME}_feats/')
        feat_dir_list[i] = feat_dir 
        if not os.path.exists(feat_dir):
            os.makedirs(feat_dir)
    print('image dirs:', img_dir_list)
    print('feat dirs:', feat_dir_list)
  
    imgs_file_list = []
    feats_file_list = []
    for i, img_dir in enumerate(img_dir_list):
        imgs_file_name = os.listdir(img_dir)
        for img_file_name in imgs_file_name:
            if not img_file_name.endswith('.jpg'):
                continue
            imgs_file_list.append(os.path.join(img_dir, img_file_name))
            feats_file_list.append(os.path.join(feat_dir_list[i], img_file_name).replace('.jpg', '.npz'))


    print('total images:', len(imgs_file_list))

    # load model
    clip_model, _ = clip.load(CLIP_MODEL, device='cpu')
    img_encoder = clip_model.visual

    model = ExtractModel(img_encoder)
    T = _transform(IMG_RESOLUTION)

    for i, img_path in tqdm(enumerate(imgs_file_list), total=len(imgs_file_list)):
        save_path = feats_file_list[i]
        if os.path.exists(save_path):
            continue
        _extract_feat(img_path, model, T, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Tool for extracting CLIP image features.')
    parser.add_argument('--config', dest='config', help='path to config file', type=str, required=True)
    # parser.add_argument('--dataset', dest='dataset', help='dataset name, e.g., ok, aok', type=str, required=True)
    # parser.add_argument('--gpu', dest='GPU', help='gpu id', type=str, default='0')
    # parser.add_argument('--clip_model', dest='CLIP_VERSION', help='clip model name or local model checkpoint path', type=str, default='RN50x64')
    # parser.add_argument('--img_resolution', dest='IMG_RESOLUTION', help='image resolution', type=int, default=512)
    args = parser.parse_args()
    args_dict = OmegaConf.load(args.config)
    for key in args_dict.keys():
        if key.endswith('_CONFIG'):
            temp_dict = OmegaConf.load(args_dict[key]) 
            args_dict = OmegaConf.merge(args_dict, temp_dict)

    main(**args_dict)