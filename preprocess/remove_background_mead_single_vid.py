import os
import numpy as np
import cv2
import glob
import json


from PIL import Image
from collections import defaultdict


# source deactivate
# source activate splatting



def convert_png_to_jpg_with_alpha(png_path, jpg_path):
    # 打开PNG图像
    png_img = Image.open(png_path)

    png_array = np.array(png_img)

    jpg_array = np.full(png_array.shape[:2] + (3,), 255, dtype=np.uint8)
    
    # 使用NumPy的广播功能，根据不透明度设置颜色
    jpg_array[...,:3][png_array[...,3] != 0] = [0, 0, 0]

    # 创建PIL图像对象
    jpg_image = Image.fromarray(jpg_array)

    # 保存结果为JPG图像
    jpg_image.save(jpg_path)

    # # 创建一个新的白色背景图像
    # new_img = Image.new("RGB", img.size, (255, 255, 255))
    
    # # 遍历每个像素并根据不透明度设置颜色
    # for y in range(height):
    #     for x in range(width):
    #         # 获取当前像素的RGBA值
    #         r, g, b, alpha = png_image.getpixel((x, y))
            
    #         # 如果不透明度不是0，将颜色设置为黑色
    #         if alpha != 0:
    #             jpg_image.putpixel((x, y), (0, 0, 0))
    
    # # 将PNG图像的alpha通道作为蒙版，将不透明部分填充为黑色
    # new_img.paste(img, mask=img.split()[3])
    
    # # 保存为JPEG格式
    # new_img.convert("RGB").save(output_path, "JPEG")
        # 打开PNG图像


def CropImage(left_up, crop_size, image=None, K=None):
    """
    对图像进行裁剪操作，并且调整内参矩阵
    """
    crop_size = np.array(crop_size).astype(np.int32)
    left_up = np.array(left_up).astype(np.int32)

    if not K is None:
        K[0:2,2] = K[0:2,2] - np.array(left_up)

    if not image is None:
        if left_up[0] < 0:
            image_left = np.zeros([image.shape[0], -left_up[0], image.shape[2]], dtype=np.uint8)
            image = np.hstack([image_left, image])
            left_up[0] = 0
        if left_up[1] < 0:
            image_up = np.zeros([-left_up[1], image.shape[1], image.shape[2]], dtype=np.uint8)
            image = np.vstack([image_up, image])
            left_up[1] = 0
        if crop_size[0] + left_up[0] > image.shape[1]:
            image_right = np.zeros([image.shape[0], crop_size[0] + left_up[0] - image.shape[1], image.shape[2]], dtype=np.uint8)
            image = np.hstack([image, image_right])
        if crop_size[1] + left_up[1] > image.shape[0]:
            image_down = np.zeros([crop_size[1] + left_up[1] - image.shape[0], image.shape[1], image.shape[2]], dtype=np.uint8)
            image = np.vstack([image, image_down])

        image = image[left_up[1]:left_up[1]+crop_size[1], left_up[0]:left_up[0]+crop_size[0], :]

    return image, K


def ResizeImage(target_size, source_size, image=None, K=None):
    """
    对图像进行resize操作，并且调整内参矩阵
    """
    if not K is None:
        K[0,:] = (target_size[0] / source_size[0]) * K[0,:]
        K[1,:] = (target_size[1] / source_size[1]) * K[1,:]

    if not image is None:
        image = cv2.resize(image, dsize=target_size)
    return image, K


def extract_masks(id_list,emos=['neutral'], views=['front']):
    for id in id_list:
        for emo in emos:
            print("Processing %s %s" % (id, emo))
            emo_path = os.path.join(DATA_SOURCE, id, emo)

            video_folders_level = os.listdir(emo_path)
            video_folders_level.sort()

            for video_folder_level in video_folders_level:
                out_folder_level = os.path.join(DATA_OUTPUT, id, emo, video_folder_level)
                
                level_path = os.path.join(emo_path, video_folder_level)
                print("Processing %s %s %s" % (id, emo, video_folder_level))
                video_folders = os.listdir(level_path)
                video_folders.sort()

                for video_folder in video_folders:
                    # 现在计数器是对每一次trial计数
                    fids = defaultdict(int)
                    print("Processing %s" % (video_folder))
                    video_paths = os.listdir(os.path.join(level_path, video_folder))
                    video_paths = [x for x in video_paths if x.endswith('.mp4')]
                    video_paths.sort()

                    out_folder_vid = os.path.join(out_folder_level, video_folder)
                    os.makedirs(out_folder_vid, exist_ok=True)

                    # 这里实际上在遍历多视角
                    for video_path in video_paths:
                        camera_id = video_path.split('.')[0].split('_')[0] 
                        mask_out_path = os.path.join(level_path, video_folder, video_path.replace('.mp4', ''))
                        # 进行转换
                        if os.path.exists(mask_out_path):
                            pass
                        else:
                            convert_video(
                                model,
                                input_source=os.path.join(level_path, video_folder, video_path),
                                output_type='png_sequence',
                                output_composition=mask_out_path
                                )
                        # ipdb.set_trace()
                        masks = os.listdir(mask_out_path)
                        masks = [mask for mask in masks if mask.endswith('.png')]
                        masks.sort()
                        # cnt=0
                        for mask in masks:
                            mask_path_png = os.path.join(mask_out_path, mask)

                            # 打开PNG图像
                            png_img = Image.open(mask_path_png)

                            png_array = np.array(png_img)

                            jpg_array = np.full(png_array.shape[:2] + (3,), 255, dtype=np.uint8)
                            
                            alpha = png_array[..., 3]
                            jpg_array[..., :3] = np.round(np.expand_dims(alpha / 255.0, axis=-1) * np.array([255, 255, 255])).astype(np.uint8)

                            # 创建PIL图像对象
                            jpg_image = Image.fromarray(jpg_array)

                            # 保存结果为JPG图像
                            jpg_image.save(os.path.join(mask_out_path, mask.replace('.png','.jpg')))

                            # 打开msak图像
                            mask_img = cv2.imread(os.path.join(mask_out_path, mask.replace('.png','.jpg')))

                            mask_img, _ = CropImage(LEFT_UP, CROP_SIZE, mask_img)
                            mask_img, _ = ResizeImage(SIZE, CROP_SIZE, mask_img)
                            
                            mask_img_lower = cv2.resize(mask_img.astype(float), SIZE_LOWRES)

                            mask_path = os.path.join(out_folder_vid, 'images', '%04d' % fids[camera_id], 'mask_' + camera_id + '.jpg')
                            mask_lowres_path = os.path.join(out_folder_vid, 'images', '%04d' % fids[camera_id], 'mask_lowres_' + camera_id + '.jpg')

                            cv2.imwrite(mask_path, mask_img)
                            cv2.imwrite(mask_lowres_path, mask_img_lower)

                            # cnt+=1
                            fids[camera_id] += 1


                    


if __name__ == "__main__":
    import ipdb
    # 确保裁切后画面中心不动
    LEFT_UP = [(1920-1080)//2, 0]
    CROP_SIZE = [1080, 1080]

    # 超参数不改
    # SIZE = [1080, 1080]
    SIZE = [2048, 2048]
    SIZE_LOWRES = [256, 256]
    # 

    DATA_SOURCE = '/data/chenziang/codes/Multiview-3DMM-Fitting/MEAD'
    DATA_OUTPUT = '../MEAD_MONO_single_vid'


    # loading video matting
    import sys
    PATH = '/data/chenziang/codes/Multiview-3DMM-Fitting/submodules/RobustVideoMatting'
    sys.path.append(PATH)   
    import segmentation_api
    from inference import convert_video
    model = segmentation_api.load_model()
    
    if not os.path.exists(DATA_OUTPUT):
        os.makedirs(DATA_OUTPUT, exist_ok=True)

    import ipdb
    # emos = ['neutral', 'angry', 'contempt', 'disgusted', 'fear', 'happy', 'sad', 'surprised']
    # emos = ['neutral']
    emos = ['angry', 'contempt', 'disgusted', 'fear', 'happy', 'sad', 'surprised']
    views = ['front']
    # ipdb.set_trace() 

    extract_masks(['M005'], emos, views)
    
