import sys

PATH = '/data/chenziang/codes/Multiview-3DMM-Fitting/submodules/face-parsing.PyTorch'
sys.path.append(PATH) 

import os
import torch
import tqdm
import glob
import numpy as np
import cv2
import argparse

import re
from config.config import config

from model import BiSeNet
import torchvision.transforms as transforms
from PIL import Image

import ipdb

# 加载模型
n_classes = 19
net = BiSeNet(n_classes=n_classes)
net.cuda()
save_pth = os.path.join('/data/chenziang/codes/Multiview-3DMM-Fitting/submodules/face-parsing.PyTorch/res/cp/79999_iter.pth')
net.load_state_dict(torch.load(save_pth))
net.eval()

to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

def get_masked_img(img_folder):
    frames = sorted(os.listdir(img_folder))
    pattern = r'image_\d+\.jpg'
    for frame in frames:
        img_frame_folder = os.path.join(img_folder, frame)
        imgs_path = sorted(os.listdir(img_frame_folder))

        imgs_raw_path = sorted([f for f in imgs_path if re.match(pattern, f)])

        # ipdb.set_trace()
        for img_raw_path in imgs_raw_path:
            # img_raw_path = os.path.join(img_frame_folder, img_raw_path)
            # mask_raw_path = os.path.join(img_frame_folder, img_raw_path.replace('image', 'mask'))

            # img_lowres_path = os.path.join(img_frame_folder, img_raw_path.replace('image', 'image_lowres'))
            # mask_lowres_path = os.path.join(img_frame_folder, img_raw_path.replace('image', 'mask_lowres'))

            img_raw = cv2.imread(os.path.join(img_frame_folder, img_raw_path)).astype(np.float32)
            mask_raw = cv2.imread(os.path.join(img_frame_folder, img_raw_path.replace('image', 'mask'))).astype(np.float32)
            mask_raw = mask_raw[:, :, 0:1]
            img_raw /= 255
            mask_raw /= 255

            out_raw = img_raw * mask_raw + (1 - mask_raw)
            out_raw *= 255

            cv2.imwrite(os.path.join(img_frame_folder, img_raw_path.replace('image', 'masked')), out_raw.astype(np.uint8))

            img_lowres = cv2.imread(os.path.join(img_frame_folder, img_raw_path.replace('image', 'image_lowres'))).astype(np.float32)
            mask_lowres = cv2.imread(os.path.join(img_frame_folder, img_raw_path.replace('image', 'mask_lowres'))).astype(np.float32)
            mask_lowres = mask_lowres[:, :, 0:1]
            img_lowres /= 255
            mask_lowres /= 255

            out_lowres = img_lowres * mask_lowres + (1 - mask_lowres)
            out_lowres *= 255

            cv2.imwrite(os.path.join(img_frame_folder, img_raw_path.replace('image', 'masked_lowres')), out_lowres.astype(np.uint8))


def vis_parsing_maps(im, parsing_anno, stride, save_im=True, save_path='face_parsing', save_path_color='face_map'):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]
    
    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    # Save result or not
    if save_im:
        # ipdb.set_trace()
        # cv2.imwrite(save_path, cv2.resize(vis_parsing_anno_color, original_size, interpolation=cv2.INTER_NEAREST))
        cv2.imwrite(save_path, vis_parsing_anno_color)
        cv2.imwrite(save_path_color, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])



def get_parsing_mask(img_folder):
    frames = sorted(os.listdir(img_folder))
    pattern = r'masked_\d+\.jpg'

    for frame in frames:
        img_frame_folder = os.path.join(img_folder, frame)
        imgs_path = sorted(os.listdir(img_frame_folder))

        imgs_raw_path = sorted([f for f in imgs_path if re.match(pattern, f)])

        for img_raw_path in imgs_raw_path:
            with torch.no_grad():
                img = Image.open(os.path.join(img_frame_folder, img_raw_path))
                size = img.size

                image = img.resize((512, 512), Image.BILINEAR)
                # image = img

                img = to_tensor(image)
                img = torch.unsqueeze(img, 0)
                img = img.cuda()
                out = net(img)[0]
                parsing = out.squeeze(0).cpu().numpy().argmax(0)

                vis_parsing_maps(
                    image, 
                    parsing, 
                    stride=1, 
                    save_im=True, 
                    save_path=os.path.join(img_frame_folder, img_raw_path.replace('masked', 'face_parsing').replace('jpg', 'png')), 
                    save_path_color=os.path.join(img_frame_folder, img_raw_path.replace('masked', 'face_map').replace('jpg', 'png')),
                    # original_size=size
                )


def get_area(mask_image, selected_labels=[], part_colors=[]):
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                [255, 0, 85], [255, 0, 170],
                [0, 255, 0], [85, 255, 0], [170, 255, 0],
                [0, 255, 85], [0, 255, 170],
                [0, 0, 255], [85, 0, 255], [170, 0, 255],
                [0, 85, 255], [0, 170, 255],
                [255, 255, 0], [255, 255, 85], [255, 255, 170],
                [255, 0, 255], [255, 85, 255], [255, 170, 255],
                [0, 255, 255], [85, 255, 255], [170, 255, 255]]
    
    result_image = np.zeros_like(mask_image)

    for label in selected_labels:
        # print(label)
        r,g,b=part_colors[label]
        selected_part = (mask_image[:, :, 0] == r) & (mask_image[:, :, 1] == g) & (mask_image[:, :, 2] == b)
        result_image[selected_part] = [255, 255, 255]


        # cv2.imwrite("mask_area.png", selected_part)
        # cv2.imwrite("result.png", result_image)
        # exit()

    return result_image


def remove_torso(img_folder):
    frames = sorted(os.listdir(img_folder))
    pattern = r'masked_\d+\.jpg'
    for frame in frames:
        img_frame_folder = os.path.join(img_folder, frame)
        imgs_path = sorted(os.listdir(img_frame_folder))

        imgs_raw_path = sorted([f for f in imgs_path if re.match(pattern, f)])
        
        # ipdb.set_trace()
        for img_raw_path in imgs_raw_path:
            raw_image = cv2.imread(os.path.join(img_frame_folder, img_raw_path))
            parsing_image = cv2.imread(os.path.join(img_frame_folder, img_raw_path.replace('masked', 'face_parsing').replace('jpg', 'png')))
            mask_area = get_area(parsing_image, selected_labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17])

            # mask_area resize 到 raw_image 大小
            mask_area = cv2.resize(mask_area, (raw_image.shape[1], raw_image.shape[0]), interpolation=cv2.INTER_NEAREST)
            # alpha = mask_area[..., 3]

            # 提取mask_area中[255,255,255]部分的索引
            mask_area = (mask_area[:, :, 0] == 255) & (mask_area[:, :, 1] == 255) & (mask_area[:, :, 2] == 255)

            # 只保留raw_image中mask_area的部分
            result_image = np.zeros((raw_image.shape[0], raw_image.shape[1], 4), dtype=np.uint8)

            # 将选定区域的像素值存储在 result_image 中，并将 alpha 通道设置为不透明
            result_image[mask_area, :3] = raw_image[mask_area]
            result_image[mask_area, 3] = 255

            # 将未被掩膜覆盖的部分的 alpha 通道设置为透明
            result_image[~mask_area, 3] = 0

            # 创建一张与 raw_image 相同大小的全白图像
            white_image = np.ones_like(raw_image) * 255

            # 将 result_image 覆盖在白色图像上
            final_result = np.where(result_image[..., 3][:, :, None] > 0, result_image[..., :3], white_image)

            # save
            cv2.imwrite(os.path.join(img_frame_folder, img_raw_path.replace('masked', 'face')), final_result)


if __name__ == '__main__':
    import ipdb
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/MEAD_M003_single_vid.yaml')
    arg = parser.parse_args()

    cfg = config()
    cfg.load(arg.config)
    cfg = cfg.get_cfg()

    device = torch.device('cuda:%d' % cfg.gpu_id)
    torch.cuda.set_device(cfg.gpu_id)

    source_folder = cfg.data_source
    print('processing %s' % source_folder)

    # ipdb.set_trace()

    emo_folders = sorted(os.listdir(source_folder))
    for emo_folder in emo_folders:
        if not os.path.isdir(os.path.join(source_folder, emo_folder)):
            continue
        print('processing %s' % emo_folder)
        level_folders = sorted(os.listdir(os.path.join(source_folder, emo_folder)))
        for level_folder in level_folders:
            print('processing %s' % level_folder)
            trial_folders = sorted(os.listdir(os.path.join(source_folder, emo_folder, level_folder)))
            for trial_folder in trial_folders:
                print('processing %s' % trial_folder)
                _source_folder = os.path.join(source_folder, emo_folder, level_folder, trial_folder, 'images')
                print('mask')
                get_masked_img(_source_folder)
                print('parsing')
                get_parsing_mask(_source_folder)
                print('remove torso')
                remove_torso(_source_folder)
                # exit()


