import torch
import numpy as np
import glob
import os
import random
import cv2
from skimage import io

class LandmarkDataset():

    def __init__(self, landmark_folder, camera_folder):
        # frames 是0000-xxxx的文件夹
        self.frames = sorted(os.listdir(landmark_folder))
        self.landmark_folder = landmark_folder
        self.camera_folder = camera_folder

    def get_item(self):
        landmarks = []
        extrinsics = []
        intrinsics = []
        # 遍历每一帧
        for frame in self.frames:
            landmarks_ = []
            extrinsics_ = []
            intrinsics_ = []
            camera_ids = [item.split('_')[-1][:-4] for item in sorted(os.listdir(os.path.join(self.landmark_folder, frame)))]
            # 按相机顺序读取每一帧
            for v in range(len(camera_ids)):
                if os.path.exists(os.path.join(self.landmark_folder, frame, 'lmk_%s.npy' % camera_ids[v])):
                    landmark = np.load(os.path.join(self.landmark_folder, frame, 'lmk_%s.npy' % camera_ids[v]))
                    landmark = np.vstack([landmark[0:48], landmark[49:54], landmark[55:68]])
                    extrinsic = np.load(os.path.join(self.camera_folder, frame, 'camera_%s.npz' % camera_ids[v]))['extrinsic']
                    intrinsic = np.load(os.path.join(self.camera_folder, frame, 'camera_%s.npz' % camera_ids[v]))['intrinsic']
                else:
                    landmark = np.zeros([66, 3], dtype=np.float32)
                    extrinsic = np.ones([3, 4], dtype=np.float32)
                    intrinsic = np.ones([3, 3], dtype=np.float32)
                landmarks_.append(landmark)
                extrinsics_.append(extrinsic)
                intrinsics_.append(intrinsic)
            # 做stack操作
            # [n,68,3]
            # n是相机数量
            landmarks_ = np.stack(landmarks_)
            extrinsics_ = np.stack(extrinsics_)
            intrinsics_ = np.stack(intrinsics_)
            # 然后添加到总的list中
            landmarks.append(landmarks_)
            extrinsics.append(extrinsics_)
            intrinsics.append(intrinsics_)
        # 再做stack操作
        # [m,n,68,3]
        # m是帧数
        landmarks = np.stack(landmarks)
        extrinsics = np.stack(extrinsics)
        intrinsics = np.stack(intrinsics)

        return landmarks, extrinsics, intrinsics, self.frames
    
    def __len__(self):
        return len(self.frames)
    
    


class LandmarkDatasetGA():

    def __init__(self, landmark_folder, camera_folder, camera_ids):
        self.frames = sorted(os.listdir(landmark_folder))
        self.landmark_folder = landmark_folder
        self.camera_folder = camera_folder
        self.camera_ids = camera_ids

    def get_item(self):
        landmarks = []
        extrinsics = []
        intrinsics = []

        # 获取总帧数
        frame_num = len(self.frames)//len(self.camera_ids)

        # 遍历每一帧
        for frame_id in range(frame_num):
            # 遍历相机
            file_names = ["{:05d}_{}".format(frame_id, camera_id) for camera_id in self.camera_ids]
            landmarks_ = []
            extrinsics_ = []
            intrinsics_ = []

            for file_name in file_names:
                landmark_path = os.path.join(self.landmark_folder, "{:s}.npy".format(file_name))
                camera_path = os.path.join(self.camera_folder, "{:s}.npz".format(file_name))

                landmark = np.load(landmark_path)
                landmark = np.vstack([landmark[0:48], landmark[49:54], landmark[55:68]])
                camera = np.load(camera_path)
                extrinsic = camera['extrinsic']
                intrinsic = camera['intrinsic']

                landmarks_.append(landmark)
                extrinsics_.append(extrinsic)
                intrinsics_.append(intrinsic)
            
            landmarks.append(np.stack(landmarks_))
            extrinsics.append(np.stack(extrinsics_))
            intrinsics.append(np.stack(intrinsics_))

            # 到这里遍历完一帧了
        
        landmarks = np.stack(landmarks)
        extrinsics = np.stack(extrinsics)
        intrinsics = np.stack(intrinsics)

        return landmarks, extrinsics, intrinsics, self.frames



        # # 遍历每一帧
        # for frame in self.frames:
        #     landmarks_ = []
        #     extrinsics_ = []
        #     intrinsics_ = []
        #     camera_ids = [item.split('_')[-1][:-4] for item in sorted(os.listdir(os.path.join(self.landmark_folder, frame)))]
        #     # 按相机顺序读取每一帧
        #     for v in range(len(camera_ids)):
        #         if os.path.exists(os.path.join(self.landmark_folder, frame, 'lmk_%s.npy' % camera_ids[v])):
        #             landmark = np.load(os.path.join(self.landmark_folder, frame, 'lmk_%s.npy' % camera_ids[v]))
        #             landmark = np.vstack([landmark[0:48], landmark[49:54], landmark[55:68]])
        #             extrinsic = np.load(os.path.join(self.camera_folder, frame, 'camera_%s.npz' % camera_ids[v]))['extrinsic']
        #             intrinsic = np.load(os.path.join(self.camera_folder, frame, 'camera_%s.npz' % camera_ids[v]))['intrinsic']
        #         else:
        #             landmark = np.zeros([66, 3], dtype=np.float32)
        #             extrinsic = np.ones([3, 4], dtype=np.float32)
        #             intrinsic = np.ones([3, 3], dtype=np.float32)
        #         landmarks_.append(landmark)
        #         extrinsics_.append(extrinsic)
        #         intrinsics_.append(intrinsic)
        #     # 做stack操作
        #     # [n,68,3]
        #     # n是相机数量
        #     landmarks_ = np.stack(landmarks_)
        #     extrinsics_ = np.stack(extrinsics_)
        #     intrinsics_ = np.stack(intrinsics_)
        #     # 然后添加到总的list中
        #     landmarks.append(landmarks_)
        #     extrinsics.append(extrinsics_)
        #     intrinsics.append(intrinsics_)
        # # 再做stack操作
        # # [m,n,68,3]
        # # m是帧数
        # landmarks = np.stack(landmarks)
        # extrinsics = np.stack(extrinsics)
        # intrinsics = np.stack(intrinsics)

        # return landmarks, extrinsics, intrinsics, self.frames
    
    def __len__(self):
        return len(self.frames)//len(self.camera_ids)