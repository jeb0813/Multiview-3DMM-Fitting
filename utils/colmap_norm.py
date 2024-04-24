import os
import numpy as np

def x_rotation(w2c):
    rotate_180_x = np.array([
                        [1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1]
                    ])
    for k,v in w2c.items():
        original_matrix = np.array(v)
        # 提取旋转部分的矩阵
        rotation_matrix = original_matrix[:3, :3]

        # 将原始旋转矩阵与绕x轴旋转180°的旋转矩阵相乘
        new_rotation_matrix = np.dot(rotation_matrix, rotate_180_x)
        # 将新的旋转矩阵放回原始的外参矩阵中
        modified_matrix = original_matrix.copy()
        modified_matrix[:3, :3] = new_rotation_matrix
        w2c[k] = modified_matrix

    return w2c



def getNerfppNorm(w2c):
    """
    获取所有相机的球体中心和半径
    返回平移矩阵和半径
    """

    w2c_translated = {}
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for k,v in w2c.items():
        W2C = v
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])
    # for cam in cam_info:
    #     W2C = getWorld2View2(cam.R, cam.T)
    #     C2W = np.linalg.inv(W2C)
    #     cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal

    # 为什么
    translate = -center

    for k,v in w2c.items():
        W2C = v.copy()
        W2C[:3, 3] += translate
        w2c_translated[k] = W2C

    return w2c_translated

if __name__=="__main__":
    import ipdb

    dataset_path = '/data/chenziang/codes/Gaussian-Head-Avatar/Mead/M003/neutral'
    cameras_path = os.path.join(dataset_path, 'cameras')
    new_cameras_path = os.path.join(dataset_path, 'new_cameras')
    frames_path = os.listdir(cameras_path)
    frames_path.sort()

    for frame_path in frames_path:
        # 对应每一帧
        extrinsics = dict()
        intrinsics = dict()
        cameras = os.listdir(os.path.join(cameras_path, frame_path))
        for camera in cameras:
            arr = np.load(os.path.join(cameras_path, frame_path, camera))
            # 加一行[0,0,0,1]
            extrinsics[camera] = np.append(arr['extrinsic'], [[0,0,0,1]], axis=0)
            intrinsics[camera] = arr['intrinsic']
        # ipdb.set_trace()
        extrinsics = x_rotation(extrinsics)
        extrinsics = getNerfppNorm(extrinsics)

        new_frame_path = os.path.join(new_cameras_path, frame_path)
        os.makedirs(new_frame_path, exist_ok=True)
        for camera in cameras:
            # extrinsics[camera] 去掉最后一行

            np.savez(os.path.join(new_frame_path, camera), extrinsic=extrinsics[camera][:-1], intrinsic=intrinsics[camera])
    
    print('Done!')



        


        
