import sys

PATH = '/data/chenziang/codes/Multiview-3DMM-Fitting/submodules/face-parsing.PyTorch'
sys.path.append(PATH) 

import os
import torch
import re
import cv2
import argparse

import torchvision.transforms as transforms
import numpy as np

from PIL import Image
from model import BiSeNet

from face_parsing import vis_parsing_maps, get_area

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



def get_parsing_mask(img_folder, pattern = r'masked_\d+\.jpg'):
    imgs_path = sorted(os.listdir(img_folder))
    imgs_raw_path = sorted([f for f in imgs_path if re.match(pattern, f)])

    for img_raw_path in imgs_raw_path:
        with torch.no_grad():
            img = Image.open(os.path.join(img_folder, img_raw_path))
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
                # save_path=os.path.join(img_folder, img_raw_path.replace('masked', 'face_parsing').replace('jpg', 'png')), 
                save_path=os.path.join(img_folder, 'face_parsing_' + img_raw_path.split('.')[0] + '.png'), 
                # save_path_color=os.path.join(img_folder, img_raw_path.replace('masked', 'face_map').replace('jpg', 'png')),
                save_path_color=os.path.join(img_folder, 'face_map_' + img_raw_path.split('.')[0] + '.png'),
                # original_size=size
            )


def remove_torso(img_folder, pattern = r'masked_\d+\.jpg'):
    imgs_path = sorted(os.listdir(img_folder))
    imgs_raw_path = sorted([f for f in imgs_path if re.match(pattern, f)])

    for img_raw_path in imgs_raw_path:
        raw_image = cv2.imread(os.path.join(img_folder, img_raw_path))
        parsing_image = cv2.imread(os.path.join(img_folder, 'face_parsing_' + img_raw_path.split('.')[0] + '.png'))
        # mask_area = get_area(parsing_image, selected_labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17])
        mask_area = get_area(parsing_image, selected_labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 17])

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
        cv2.imwrite(os.path.join(img_folder, 'face_' + img_raw_path.split('.')[0] + '.jpg'), final_result)


def get_parser():
    parser = argparse.ArgumentParser(description='face_parsing')
    parser.add_argument('--source_folder', 
                        type=str, 
                        default='/data/chenziang/codes/Gaussian-Head-Avatar-private/output_mead/results/reenactment/M003_neutral', 
                        help='Path to the generated subject folder'
                    )

    return parser.parse_args()


if __name__ == '__main__':
    import ipdb
    # source_folder = '/data/chenziang/codes/Gaussian-Head-Avatar-private/output_mead/results/reenactment/M003_neutral'
    args = get_parser()
    source_folder = args.source_folder

    print('processing %s' % source_folder)

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
                _source_folder = os.path.join(source_folder, emo_folder, level_folder, trial_folder)
                
                # print('mask')
                # get_masked_img(_source_folder)
                print('parsing')
                get_parsing_mask(_source_folder, pattern = r'\d{4}\.jpg')
                print('remove torso')
                remove_torso(_source_folder, pattern = r'\d{4}\.jpg')
                # exit()