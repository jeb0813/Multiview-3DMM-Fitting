import os
import numpy as np
import cv2
import glob
import json

import argparse
from collections import defaultdict

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

def cp_vid(dataset_path, cam2id, emo, views):
    video_folder = os.path.join(dataset_path, 'video')

    out_path = os.path.join(dataset_path, emo)
    os.makedirs(out_path, exist_ok=True)

    # 遍历视角
    for view in views:
        view_folder = os.path.join(video_folder, view)
        view_emo_path = os.path.join(view_folder, emo)
        if not os.path.exists(view_emo_path):
            raise ValueError("No such path %s" % view_emo_path)
        level_folders = os.listdir(view_emo_path)
        level_folders.sort()

        # 遍历level
        for level_folder in level_folders:
            out_level_path = os.path.join(out_path, level_folder)
            os.makedirs(out_level_path, exist_ok=True)
        
            video_paths = os.listdir(os.path.join(view_emo_path, level_folder))
            video_paths.sort()

            for video_path in video_paths:
                out_trial = video_path[:3]
                out_trial_path = os.path.join(out_level_path, out_trial)
                os.makedirs(out_trial_path, exist_ok=True)

                # 复制当前视频到out_id_path，重命名cam2id[view]
                video_path = os.path.join(view_emo_path, level_folder, video_path)
                out_video_path = os.path.join(out_trial_path, cam2id[view] + '_' + view + '.mp4')
                os.system('cp %s %s' % (video_path, out_video_path))
    
    # # 处理音频
    # audio_folder = os.path.join(dataset_path, 'audio')
    # audio_emo_path = os.path.join(audio_folder, emo)
    # if not os.path.exists(audio_emo_path):
    #     raise ValueError("No such path %s" % audio_emo_path)
    # audio_level_folders = os.listdir(audio_emo_path)
    # audio_level_folders.sort()

    # for audio_level_folder in audio_level_folders:
    #     out_level_path = os.path.join(out_path, audio_level_folder)
    #     if not os.path.exists(out_level_path):
    #         raise ValueError("No such path %s" % out_level_path)
        
    #     audio_paths = os.listdir(os.path.join(audio_emo_path, audio_level_folder))
    #     audio_paths.sort()

    #     for audio_path in audio_paths:
    #         out_trial = audio_path[:3]
    #         out_trial_path = os.path.join(out_level_path, out_trial)
    #         if not os.path.exists(out_trial_path):
    #             raise ValueError("No such path %s" % out_trial_path)

    #         # 复制当前音频到out_id_path
    #         audio_path = os.path.join(audio_emo_path, audio_level_folder, audio_path)
    #         out_audio_path = os.path.join(out_trial_path, 'audio.wav')
    #         os.system('cp %s %s' % (audio_path, out_audio_path))


def extract_frames(id,emos=['neutral'], views=['front']):
    # camera_path = os.path.join(DATA_SOURCE, 'cam_params.json')
    # for id in id_list:
    # camera_path="/data/chenziang/codes/Gaussian-Head-Avatar/data_mead/cam_params.json"
    camera_path = os.path.join(DATA_SOURCE, id, 'cam_params.json')
    with open(camera_path, 'r') as f:
        camera = json.load(f)

    # 遍历emo
    for emo in emos:
        print("Processing %s %s" % (id, emo))
        emo_path = os.path.join(DATA_SOURCE, id, emo)

        # ipdb.set_trace()
        if not os.path.exists(emo_path):
            print("No such path %s" % emo_path)
            print("Extracting from video")
            # if not exist, extract
            # 一个emo 一个 level 一组
            cp_vid(os.path.join(DATA_SOURCE, id), camera["cam_name2id"], emo, views)
        # exit(0)

        
        video_folders_level = os.listdir(emo_path)
        video_folders_level.sort()

        # 遍历level
        for video_folder_level in video_folders_level:
            # fids = defaultdict(int)
            out_folder_level = os.path.join(DATA_OUTPUT, id, emo, video_folder_level)
            os.makedirs(out_folder_level, exist_ok=True)

            level_path = os.path.join(emo_path, video_folder_level)
            print("Processing %s %s %s" % (id, emo, video_folder_level))
            video_folders = os.listdir(level_path)
            video_folders.sort()

            # import ipdb
            # ipdb.set_trace()

            # 遍历video，每个vid单独一个文件夹
            for video_folder in video_folders:
                # 现在计数器是对每一次trial计数
                fids = defaultdict(int)
                print("Processing %s" % (video_folder))
                video_paths = os.listdir(os.path.join(level_path, video_folder))
                video_paths = [x for x in video_paths if x.endswith('.mp4')]
                video_paths.sort()

                out_folder_vid = os.path.join(out_folder_level, video_folder)
                os.makedirs(out_folder_vid, exist_ok=True)

                # ipdb.set_trace()
                # 这里实际上在遍历多视角
                for video_path in video_paths:
                    camera_id = video_path.split('.')[0].split('_')[0]

                    # 去掉最后一行
                    extrinsic = np.array(camera['world_2_cam'][camera_id][:3])
                    intrinsic = np.array(camera['intrinsics'][camera_id])

                    # 预处理intrinsics
                    # fx,fy,cx,cy=intrinsic['fx'],intrinsic['fy'],intrinsic['cx'],intrinsic['cy']
                    # intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                    # intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                    _, intrinsic = CropImage(LEFT_UP, CROP_SIZE, None, intrinsic)
                    _, intrinsic = ResizeImage(SIZE, CROP_SIZE, None, intrinsic)
                    
                    cap = cv2.VideoCapture(os.path.join(level_path, video_folder, video_path))

                    while(1): 
                        _, image = cap.read()
                        if image is None:
                            break

                        visible = (np.ones_like(image) * 255).astype(np.uint8)
                        image, _ = CropImage(LEFT_UP, CROP_SIZE, image, None)
                        image, _ = ResizeImage(SIZE, CROP_SIZE, image, None)
                        visible, _ = CropImage(LEFT_UP, CROP_SIZE, visible, None)
                        visible, _ = ResizeImage(SIZE, CROP_SIZE, visible, None)
                        image_lowres = cv2.resize(image, SIZE_LOWRES)
                        visible_lowres = cv2.resize(visible, SIZE_LOWRES)

                        os.makedirs(os.path.join(out_folder_vid, 'images', '%04d' % fids[camera_id]), exist_ok=True)
                        cv2.imwrite(os.path.join(out_folder_vid, 'images', '%04d' % fids[camera_id], 'image_' + camera_id + '.jpg'), image)
                        cv2.imwrite(os.path.join(out_folder_vid, 'images', '%04d' % fids[camera_id], 'image_lowres_' + camera_id + '.jpg'), image_lowres)
                        cv2.imwrite(os.path.join(out_folder_vid, 'images', '%04d' % fids[camera_id], 'visible_' + camera_id + '.jpg'), visible)
                        cv2.imwrite(os.path.join(out_folder_vid, 'images', '%04d' % fids[camera_id], 'visible_lowres_' + camera_id + '.jpg'), visible_lowres)
                        os.makedirs(os.path.join(out_folder_vid, 'cameras', '%04d' % fids[camera_id]), exist_ok=True)
                        np.savez(os.path.join(out_folder_vid, 'cameras', '%04d' % fids[camera_id], 'camera_' + camera_id + '.npz'), extrinsic=extrinsic, intrinsic=intrinsic)
                        
                        fids[camera_id] += 1


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_source', type=str, default='/data/chenziang/codes/Multiview-3DMM-Fitting/MEAD')
    parser.add_argument('--data_output', type=str, default='../MEAD_MONO_single_vid')
    parser.add_argument('--left_up', type=list, default=[(1920-1080)//2, 0])
    parser.add_argument('--crop_size', type=list, default=[1080, 1080])
    parser.add_argument('--size', type=list, default=[1080, 1080])
    parser.add_argument('--size_lowres', type=list, default=[256, 256])
    parser.add_argument('--subject_id', type=str, default='M003')
    return parser.parse_args()



if __name__ == "__main__":
    """
    单目数据
    """
    args = get_args()

    LEFT_UP = args.left_up
    CROP_SIZE = args.crop_size
    SIZE = args.size
    SIZE_LOWRES = args.size_lowres
    DATA_SOURCE = args.data_source
    DATA_OUTPUT = args.data_output
    subject_id = args.subject_id


    # 确保裁切后画面中心不动
    # LEFT_UP = [(1920-1080)//2, 0]
    # CROP_SIZE = [1080, 1080]

    # 超参数改为1080， 1080
    # SIZE = [1080, 1080]
    # SIZE = [2048, 2048]
    # SIZE_LOWRES = [256, 256]

    # DATA_SOURCE = '/data/chenziang/codes/Multiview-3DMM-Fitting/MEAD'
    # DATA_OUTPUT = '../MEAD_MONO_single_vid'

    if not os.path.exists(DATA_OUTPUT):
        os.makedirs(DATA_OUTPUT, exist_ok=True)

    import ipdb
    emos = ['neutral', 'angry', 'contempt', 'disgusted', 'fear', 'happy', 'sad', 'surprised']
    # emos = ['neutral']
    views = ['front']
    # ipdb.set_trace() 
    extract_frames(subject_id, emos, views)

