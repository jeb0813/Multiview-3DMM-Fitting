### 读取3d lmk和图像，实现投影和绘图功能
import sys
sys.path.append('/data/chenziang/codes/Multiview-3DMM-Fitting/')


import torch
import os, re
import numpy as np

from lib.Fitter import MyFitter
from lib.Camera import Camera
from PIL import Image
from lib.face_models.BFMModule import BFMModule

def get_fitter(gpu_id=0):


    device = torch.device('cuda:%d' % gpu_id)
    torch.cuda.set_device(gpu_id)

    # dataset = LandmarkDataset(landmark_folder=cfg.landmark_folder, camera_folder=cfg.camera_folder)
    # face_model = get_face_model(cfg.face_model, batch_size=len(dataset), device=device)
    camera = Camera(image_size=2048)
    # recorder = Recorder(save_folder=cfg.param_folder, camera=camera, visualize=cfg.visualize, save_vertices=cfg.save_vertices)

    fitter = MyFitter(None, None, None, camera, None, device)

    return fitter


if __name__=="__main__":
    # root_path="/data/chenziang/codes/Gaussian-Head-Avatar/NeRSemble/074"
    # root_path="/data/chenziang/codes/Gaussian-Head-Avatar/Mead/M003/neutral"
    root_path="/data/chenziang/codes/Gaussian-Head-Avatar/dummy_Mead/M003/neutral"
    out_path="/data/chenziang/codes/Gaussian-Head-Avatar/NeRSemble/temp_2"
    image_path=os.path.join(root_path,"images")
    landmark_path=os.path.join(root_path,"landmarks")
    camera_path=os.path.join(root_path,"cameras")

    face_model = BFMModule(1).to('cuda:0')

    fitter = get_fitter()


    timesteps = os.listdir(image_path)
    timesteps.sort()

    import ipdb


    for timestep in timesteps:
        print("timestep:",timestep)
        img_path=os.path.join(image_path,timestep)
        lmd_path=os.path.join(landmark_path,timestep)
        cam_path=os.path.join(camera_path,timestep)

        imgs=os.listdir(img_path)
        # print("imgs:",imgs)
        # 正则匹配<image_[0-9].jpg>
        pattern = re.compile(r"image_[0-9]+\.jpg")
        imgs = [x for x in imgs if pattern.match(x)]
        imgs.sort()



        for img in imgs:
            # 3d lmk
            # [1, 66, 3]
            _, landmarks_3d = face_model()


            cam_idx=img.split("_")[1].split(".")[0]
            img_file=os.path.join(img_path,img)
            lmd_file=os.path.join(lmd_path,"lmk_%s.npy"%cam_idx)
            cam_file=os.path.join(cam_path,"camera_%s.npz"%cam_idx)

            print("img_file:",img_file)
            print("lmd_file:",lmd_file)
            print("cam_file:",cam_file)

            # cam npz下两个字段 ['extrinsic', 'intrinsic']
            # extrinsic 形状为(3,4)
            # intrinsic 形状为(3,3)
            cam = np.load(cam_file)
            extrinsic = cam['extrinsic']
            intrinsic = cam['intrinsic']

            print("extrinsic:",extrinsic)
            print("intrinsic:",intrinsic)

            # ipdb.set_trace()


            # rotation=extrinsic[:3, :3].T
            # extrinsic=np.array([extrinsic[0],
            #                     -extrinsic[1],
            #                     -extrinsic[2]])


            # extrinsic = np.hstack((rotation, translation))
            
        #     params = [[
        #         0.9958578944206238,
        #         0.0009042018209584057,
        #         -0.09091925621032715,
        #         -0.02060595527291298
        #     ],
        #     [
        #         0.023861341178417206,
        #         -0.967501699924469,
        #         0.2517364025115967,
        #         -0.029934503138065338
        #     ],
        #     [
        #         -0.08773690462112427,
        #         -0.25286316871643066,
        #         -0.9635157585144043,
        #         1.1919126510620117
        #     ],
        #     [
        #         0.0,
        #         0.0,
        #         0.0,
        #         1.0
        #     ]]

        #     params = np.array(params)

        #     extrinsic = params[:3,:4]
        #     intrinsic = np.array([[
        #     8194.6318359375,
        #     0.0,
        #     1099.349365234375
        # ],
        # [
        #     0.0,
        #     8194.7333984375,
        #     1603.513916015625
        # ],
        # [
        #     0.0,
        #     0.0,
        #     1.0
        # ]])

            
            

            # # opencv -> opengl 转换
            # # x轴相同，yz轴反向
            # extrinsic[1] = -extrinsic[1]
            # extrinsic[2] = -extrinsic[2]
            # # 旋转矩阵转置
            # extrinsic[:3,:3] = extrinsic[:3,:3].T
            # # 旋转矩阵逆
            # extrinsic[:3,:3] = np.linalg.inv(extrinsic[:3,:3])
            # # 平移矩阵逆
            # extrinsic[:3,3] = -np.dot(extrinsic[:3,:3],extrinsic[:3,3])
            # ###


            # landmarks_3d = torch.from_numpy(landmarks_3d).float()
            # extrinsic intrinsic 增加batch维度
            extrinsic = torch.from_numpy(extrinsic).unsqueeze(0).float()
            intrinsic = torch.from_numpy(intrinsic).unsqueeze(0).float()

            # 移动到gpu
            extrinsic = extrinsic.to('cuda:0')
            intrinsic = intrinsic.to('cuda:0')
            # extrinsic = torch.from_numpy(extrinsic).float()
            # intrinsic = torch.from_numpy(intrinsic).float()
            ipdb.set_trace()
            landmarks_2d = fitter.project(landmarks_3d, intrinsic, extrinsic)




            # lmd 形状为(68,3)
            lmk_gt = np.load(lmd_file)
            # 取前两维
            lmk_gt = lmk_gt[:,:2]
            # 加载图片
            image = Image.open(img_file)
            print(lmk_gt.shape)
            print(lmk_gt)

            # 将lmd绘制到图片上
            for l in lmk_gt:
                x, y = l
                x = int(x)
                y = int(y)
                for i in range(-3, 4):
                    for j in range(-3, 4):
                        image.putpixel((x + i, y + j), (255, 0, 0))

            # 将landmarks_2d绘制到图片上
            # tensor 2 numpy
            landmarks_2d = landmarks_2d[0].cpu().detach().numpy()

            # ###
            # # landmarks_2d 归一化到【-1,1】
            # landmarks_2d  = 2 * (landmarks_2d  - np.min(landmarks_2d )) / (np.max(landmarks_2d ) - np.min(landmarks_2d )) -1
            # # 放缩到【48，2000】
            # landmarks_2d  = landmarks_2d * ((2000 - 48)/2)
            # landmarks_2d  = landmarks_2d + (2000 + 48)/2

            # ###

            print(landmarks_2d.shape)
            print(landmarks_2d)
            exit()

            for l in landmarks_2d:
                x, y = l
                x = int(x)
                y = int(y)
                for i in range(-3, 4):
                    for j in range(-3, 4):
                        image.putpixel((x + i, y + j), (0, 255, 0))
            
            # 保存图片
            image.save(os.path.join(out_path,img))






            # print(lmk_3d.shape)
            # print(lmk_3d)
            # exit()

            # # cam npz下两个字段 ['extrinsic', 'intrinsic']
            # # extrinsic 形状为(3,4)
            # # intrinsic 形状为(3,3)
            # cam = np.load(cam_file)
            # extrinsic = cam['extrinsic']
            # intrinsic = cam['intrinsic']

            # # 增加batch维度
            # lmk_3d = torch.from_numpy(lmk_3d).unsqueeze(0).float()
            # intrinsic = torch.from_numpy(intrinsic).unsqueeze(0).float()
            # extrinsic = torch.from_numpy(extrinsic).unsqueeze(0).float()


            # # call fitter.project
            # landmarks_2d = fitter.project(lmk_3d, intrinsic, extrinsic)
            
            # # 归一化
            # landmarks_2d = landmarks_2d / 2048

            # print(landmarks_2d.shape)
            # print(landmarks_2d)

            # # landmarks_2d = fitter.project(landmarks_3d, intrinsics, extrinsics)
                


            # exit()