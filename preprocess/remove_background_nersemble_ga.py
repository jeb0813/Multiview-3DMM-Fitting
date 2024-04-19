import argparse
import torch
import os
import shutil
import json
import glob
import cv2
import imageio
import re
import numpy as np
from tqdm import tqdm

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.transforms.functional import to_pil_image
from threading import Thread

from dataset import ImagesDataset, ZipDataset
from dataset import augmentation as A
from model import MattingBase, MattingRefine
from inference_utils import HomographicAlignment

cam2id={
    0:'cam_222200042',
    1:'cam_222200044',
    2:'cam_222200046',
    3:'cam_222200040',
    4:'cam_222200036',
    5:'cam_222200048',
    6:'cam_220700191',
    7:'cam_222200041',
    8:'cam_222200037',
    9:'cam_222200038',
    10:'cam_222200047',
    11:'cam_222200043',
    12:'cam_222200049',
    13:'cam_222200039',
    14:'cam_222200045',
    15:'cam_221501007',
}



def preprocess_nersemble(args, data_folder, camera_ids):
    # 遍历相机
    imgs_path = os.path.join(data_folder, 'images')
    imgs = os.listdir(imgs_path)

    masks_path = os.path.join(data_folder, 'fg_masks')

    pattern_template = r'\d{5}_{:s}\.png'
    # 正则匹配
    for cam_id in camera_ids:
        background_path = os.path.join(data_folder, 'background', 'bg_%s.jpg' % cam2id[int(cam_id)][:4])
        # 构建表达式
        pattern = pattern_template.format(cam_id)
        # 匹配
        img_list = [img for img in imgs if re.match(pattern, img)]
        for img in tqdm(img_list):
            img_path = os.path.join(imgs_path, img)
            image = imageio.imread(img_path)
            src = (torch.from_numpy(image).float() / 255).permute(2,0,1)[None].to(device, non_blocking=True)

            if os.path.exists(background_path):
                background = imageio.imread(background_path)
                bgr = (torch.from_numpy(background).float() / 255).permute(2,0,1)[None].to(device, non_blocking=True)
            else:
                bgr = src * 0.0
                
            with torch.no_grad():
                if args.model_type == 'mattingbase':
                    pha, fgr, err, _ = model(src, bgr)
                elif args.model_type == 'mattingrefine':
                    pha, fgr, _, _, err, ref = model(src, bgr)
            mask = (pha[0].repeat([3, 1, 1]) * 255).permute(1,2,0).cpu().numpy().astype(np.uint8)


            mask_path = os.path.join(masks_path, img)
            imageio.imsave(mask_path, mask)






    # fids = sorted(os.listdir(os.path.join(data_folder, 'images')))
    # for v in range(len(camera_ids)):
    #     for fid in tqdm(fids):
    #         image_path = os.path.join(data_folder, 'images', fid, 'image_%s.jpg' % camera_ids[v])
    #         background_path = os.path.join(data_folder, 'background', 'image_%s.jpg' % camera_ids[v])
    #         if not os.path.exists(image_path):
    #             continue
    #         image = imageio.imread(image_path)
    #         src = (torch.from_numpy(image).float() / 255).permute(2,0,1)[None].to(device, non_blocking=True)

    #         if os.path.exists(background_path):
    #             background = imageio.imread(background_path)
    #             bgr = (torch.from_numpy(background).float() / 255).permute(2,0,1)[None].to(device, non_blocking=True)
    #         else:
    #             bgr = src * 0.0
                
    #         with torch.no_grad():
    #             if args.model_type == 'mattingbase':
    #                 pha, fgr, err, _ = model(src, bgr)
    #             elif args.model_type == 'mattingrefine':
    #                 pha, fgr, _, _, err, ref = model(src, bgr)
    #         mask = (pha[0].repeat([3, 1, 1]) * 255).permute(1,2,0).cpu().numpy().astype(np.uint8)
    #         mask_lowres = cv2.resize(mask, (256, 256))

    #         mask_path = os.path.join(data_folder, 'images', fid, 'mask_%s.jpg' % camera_ids[v])
    #         imageio.imsave(mask_path, mask)

    #         mask_lowres_path = os.path.join(data_folder, 'images', fid, 'mask_lowres_%s.jpg' % camera_ids[v])
    #         imageio.imsave(mask_lowres_path, mask_lowres)


if __name__ == "__main__":
    import ipdb
    parser = argparse.ArgumentParser(description='Inference images')

    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--model-type', type=str, default='mattingrefine', choices=['mattingbase', 'mattingrefine'])
    parser.add_argument('--model-backbone', type=str, default='resnet101', choices=['resnet101', 'resnet50', 'mobilenetv2'])
    parser.add_argument('--model-backbone-scale', type=float, default=0.25)
    parser.add_argument('--model-checkpoint', type=str, default='assets/pytorch_resnet101.pth')
    parser.add_argument('--model-refine-mode', type=str, default='sampling', choices=['full', 'sampling', 'thresholding'])
    parser.add_argument('--model-refine-sample-pixels', type=int, default=80_000)
    parser.add_argument('--model-refine-threshold', type=float, default=0.7)
    parser.add_argument('--model-refine-kernel-size', type=int, default=3)
    args = parser.parse_args()

    # 初始化模型
    device = torch.device(args.device)
    # Load model
    if args.model_type == 'mattingbase':
        model = MattingBase(args.model_backbone)
    if args.model_type == 'mattingrefine':
        model = MattingRefine(
            args.model_backbone,
            args.model_backbone_scale,
            args.model_refine_mode,
            args.model_refine_sample_pixels,
            args.model_refine_threshold,
            args.model_refine_kernel_size)

    model = model.to(device).eval()
    model.load_state_dict(torch.load(args.model_checkpoint, map_location=device), strict=False)

    ipdb.set_trace()

    DATA_SOURCE = '/data/chenziang/codes/Multiview-3DMM-Fitting/NeRSemble_ga'
    # CAMERA_IDS = ['220700191', '221501007', '222200036', '222200037', '222200038', '222200039', '222200040', '222200041',
    #               '222200042', '222200043', '222200044', '222200045', '222200046', '222200047', '222200048', '222200049']

    CAMERA_IDS = ['{:02d}'.format(i) for i in range(16)]

    trials = sorted(os.listdir(DATA_SOURCE))
    for trial in trials:
        if not (trial.startswith('EMO') or trial.startswith('EXP')):
            continue
        print("Processing trial: ", trial)
        data_folder = os.path.join(DATA_SOURCE, trial)
        preprocess_nersemble(args, data_folder, CAMERA_IDS)