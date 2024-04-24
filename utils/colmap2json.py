"""
将colmap输出的txt转写成json文件
"""

import argparse
import json

import numpy as np

def parser_args():
    parser = argparse.ArgumentParser(description='Convert colmap txt to json')
    parser.add_argument('--cameras_path', type=str, default='cameras.txt', help='colmap txt file path')
    parser.add_argument('--images_txt', type=str, default='images.txt', help='colmap txt file path')
    parser.add_argument('--json_path', type=str, default='camera_params.json', help='json file path')
    args = parser.parse_args()
    return args

def read_txt(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()


    return lines

def get_images_params(images):
    #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    images_params = dict()
    cam2id = dict()

    for image in images:
        image = image.strip().split()   
        camera_id = image[8]
        name = image[9].split('.')[0].split('_')[-1]
        cam2id[name] = camera_id

        qw, qx, qy, qz = image[1:5]
        tx, ty, tz = image[5:8]
        # str,convert to float and to ndarray
        arr = [qw, qx, qy, qz, tx, ty, tz]
        arr = list(map(float, arr))
        arr = np.array(arr)
        images_params[camera_id] = arr
    
    return images_params, cam2id

def get_camera_params(cameras):
    #   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
    camera_params = dict()
    for camera in cameras:
        camera = camera.strip().split()
        camera_id = camera[0]
        model = camera[1]
        width = camera[2]
        height = camera[3]
        params = camera[4:]

        # f, cx, cy, k
        # str,convert to float and to ndarray
        params = list(map(float, params))
        params = np.array(params)
        camera_params[camera_id] = params
    return camera_params


def quaternion2rot(qw, qx, qy, qz):
    """
    Convert quaternion to rotation matrix.
    """

    return np.array([[1.0 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
                     [2*qx*qy + 2*qz*qw, 1.0 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
                     [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1.0 - 2*qx**2 - 2*qy**2]])

def quaternion2mat(qw, qx, qy, qz, tx, ty, tz):
    """
    Convert quaternion and translation to 4x4 transformation matrix.
    """
    rotation_matrix = quaternion2rot(qw, qx, qy, qz)
    return np.vstack((np.hstack((rotation_matrix, np.array([[tx], [ty], [tz]]))),
                      np.array([0, 0, 0, 1])))


def get_intrinsic_mat(f,cx,cy,k):
    return [[f, 0, cx],
            [0, f, cy],
            [0, 0, 1]
            ]



def get_params(cam2id, images_params, camera_params):
    # params = dict()
    world_2_cam = dict()
    intrinsics = dict()

    for cam_name, cam_id in cam2id.items():
    # for cam_id in cam2id.values():
        camera_param = camera_params[cam_id]
        image_param = images_params[cam_id]

        extrinsic = quaternion2mat(*image_param)
        intrinsic = get_intrinsic_mat(*camera_param)

        world_2_cam[cam_name] = extrinsic
        intrinsics[cam_name] = intrinsic
    
    return world_2_cam, intrinsics



if __name__=="__main__":
    args = parser_args()
    cameras_path = args.cameras_path
    images_txt = args.images_txt
    json_path = args.json_path

    cameras = read_txt(cameras_path)
    images = read_txt(images_txt)

    # cameras 前三行是注释
    cameras = cameras[3:]
    # images 前四行是注释
    images = images[4:]
    # 每两行是一组数据
    images = images[::2]

    assert len(cameras) == len(images)

    images_params, cam2id = get_images_params(images)
    camera_params = get_camera_params(cameras)

    # 所有的变换在这里做
    world_2_cam, intrinsics = get_params(cam2id, images_params, camera_params)
    # ndarray to str list in order to save in json
    for key in world_2_cam.keys():
        world_2_cam[key] = world_2_cam[key].tolist()
    # for key in intrinsics.keys():
    #     intrinsics[key] = intrinsics[key].tolist()


    # 写入json
    with open(json_path, 'w') as f:
        json.dump({'world_2_cam': world_2_cam, 'intrinsics': intrinsics}, f)




    