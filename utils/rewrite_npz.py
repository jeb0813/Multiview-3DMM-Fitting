import os 
import numpy as np

cams_path = '/data/chenziang/codes/Multiview-3DMM-Fitting/MEAD_MONO/M003/neutral/level_1/cameras'

cams_folders = os.listdir(cams_path)

for cam_folder in cams_folders:
    cam_path = os.path.join(cams_path, cam_folder)
    cam_files = os.listdir(cam_path)
    for cam_file in cam_files:
        if cam_file.endswith('.npz'):
            cam_file_path = os.path.join(cam_path, cam_file)
            arr = np.load(cam_file_path)
            extrinsic = arr['extrinsic']
            intrinsic = arr['intrinsic']
            extrinsic[1][1] = -1
            extrinsic[2][2] = -1
            extrinsic[2][3] = 2

            np.savez(cam_file_path, extrinsic=extrinsic, intrinsic=intrinsic)

