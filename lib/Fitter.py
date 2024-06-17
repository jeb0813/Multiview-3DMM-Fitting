import torch
import numpy as np
from tqdm import tqdm
from einops import rearrange


class Fitter():
    def __init__(self, cfg, dataset, face_model, camera, recorder, device):
        self.cfg = cfg
        self.dataset = dataset
        self.face_model = face_model
        self.camera = camera
        self.recorder = recorder
        self.device = device

        self.optimizers = [torch.optim.Adam([{'params' : self.face_model.scale, 'lr' : 1e-3},
                                             {'params' : self.face_model.pose, 'lr' : 1e-2}]),
                                             #{'params' : self.face_model.translation, 'lr' : 1e-2}]),
                           torch.optim.Adam([{'params' : self.face_model.parameters(), 'lr' : 1e-3}])]
    
    def run(self):
        print("loading data")
        landmarks_gt, extrinsics0, intrinsics0, frames = self.dataset.get_item()
        # [2706, 16, 66, 3] 16是机位数量
        landmarks_gt = torch.from_numpy(landmarks_gt).float().to(self.device)
        extrinsics0 = torch.from_numpy(extrinsics0).float().to(self.device)
        intrinsics0 = torch.from_numpy(intrinsics0).float().to(self.device)
        extrinsics = rearrange(extrinsics0, 'b v x y -> (b v) x y')
        intrinsics = rearrange(intrinsics0, 'b v x y -> (b v) x y')
        
        print('Start fitting')
        for optimizer in self.optimizers:
            pprev_loss = 1e8
            prev_loss = 1e8

            # 创建进度条对象
            progress_bar = tqdm(range(int(1e10)))

            # for i in range(int(1e10)):
            for i in progress_bar:
                # landmarks_3d: [2706, 66, 3]
                # 每一帧对应一个[66, 3]
                _, landmarks_3d = self.face_model()

                # import ipdb
                # ipdb.set_trace()

                # [2706, 66, 3] -> [2706, 16, 66, 3]
                landmarks_3d = landmarks_3d.unsqueeze(1).repeat(1, landmarks_gt.shape[1], 1, 1)
                # [2706, 16, 66, 3] -> [43296, 66, 3]
                landmarks_3d = rearrange(landmarks_3d, 'b v x y -> (b v) x y')

                # [43296, 66, 2]
                landmarks_2d = self.project(landmarks_3d, intrinsics, extrinsics)
                # [43296, 66, 2] -> [2706, 16, 66, 2]
                landmarks_2d = rearrange(landmarks_2d, '(b v) x y -> b v x y', b=landmarks_gt.shape[0])

                pro_loss = (((landmarks_2d / self.cfg.image_size - landmarks_gt[:, :, :, 0:2] / self.cfg.image_size) * landmarks_gt[:, :, :, 2:3]) ** 2).sum(-1).sum(-2).mean()
                reg_loss = self.face_model.reg_loss(self.cfg.reg_id_weight, self.cfg.reg_exp_weight)
                loss = pro_loss + reg_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 更新进度条
                progress_bar.set_postfix(loss=loss.item())

                if abs(loss.item() - prev_loss) < 1e-10 and abs(loss.item() - pprev_loss) < 1e-9:
                # if abs(loss.item() - prev_loss) < 1e-7 and abs(loss.item() - pprev_loss) < 1e-6:
                    break
                else:
                    pprev_loss = prev_loss
                    prev_loss = loss.item()
        
        # 关闭进度条
        progress_bar.close()

        log = {
            # all frame folders
            'frames': frames,
            # [2706, 16, 66, 3]
            'landmarks_gt': landmarks_gt,
            # [2706, 16, 66, 2]
            'landmarks_2d': landmarks_2d.detach(),
            'face_model': self.face_model,
            'intrinsics': intrinsics0,
            'extrinsics': extrinsics0
        }
        print('writing')
        self.recorder.log(log)


    def project(self, points_3d, intrinsic, extrinsic):
        points_3d = points_3d.permute(0,2,1)
        calibrations = torch.bmm(intrinsic, extrinsic)
        points_2d = self.camera.project(points_3d, calibrations)
        points_2d = points_2d.permute(0,2,1)
        return points_2d
    

class MyFitter(Fitter):
    def __init__(self, cfg, dataset, face_model, camera, recorder, device):
        self.cfg = cfg
        self.dataset = dataset
        self.face_model = face_model
        self.camera = camera
        self.recorder = recorder
        self.device = device

        #self.optimizers = [torch.optim.Adam([{'params' : self.face_model.scale, 'lr' : 1e-3},
        #                                     {'params' : self.face_model.pose, 'lr' : 1e-2}]),
        #                   torch.optim.Adam([{'params' : self.face_model.parameters(), 'lr' : 1e-3}])]
    
    def run(self):
        print("loading data")
        landmarks_gt, extrinsics0, intrinsics0, frames = self.dataset.get_item()
        # (180, 16, 66, 3) 16是机位数量
        landmarks_gt = torch.from_numpy(landmarks_gt).float().to(self.device)
        # (180, 16, 3, 4)
        extrinsics0 = torch.from_numpy(extrinsics0).float().to(self.device)
        # (180, 16, 3, 3)
        intrinsics0 = torch.from_numpy(intrinsics0).float().to(self.device)
        # torch.Size([2880, 3, 4])
        extrinsics = rearrange(extrinsics0, 'b v x y -> (b v) x y')
        # torch.Size([2880, 3, 3])
        intrinsics = rearrange(intrinsics0, 'b v x y -> (b v) x y')
        
        print('Start fitting')
        for optimizer in self.optimizers:
            pprev_loss = 1e8
            prev_loss = 1e8

            # 创建进度条对象
            progress_bar = tqdm(range(int(1e10)))

            # for i in range(int(1e10)):
            for i in progress_bar:
                # landmarks_3d: [2706, 66, 3]
                # 每一帧对应一个[66, 3]
                _, landmarks_3d = self.face_model()

                # import ipdb
                # ipdb.set_trace()

                # [2706, 66, 3] -> [2706, 16, 66, 3]
                landmarks_3d = landmarks_3d.unsqueeze(1).repeat(1, landmarks_gt.shape[1], 1, 1)
                # [2706, 16, 66, 3] -> [43296, 66, 3]
                landmarks_3d = rearrange(landmarks_3d, 'b v x y -> (b v) x y')

                # [43296, 66, 2]
                landmarks_2d = self.project(landmarks_3d, intrinsics, extrinsics)
                # [43296, 66, 2] -> [2706, 16, 66, 2]
                landmarks_2d = rearrange(landmarks_2d, '(b v) x y -> b v x y', b=landmarks_gt.shape[0])

                pro_loss = (((landmarks_2d / torch.tensor([self.cfg.image_size_x, self.cfg.image_size_y], device=self.cfg.gpu_id) - landmarks_gt[:, :, :, 0:2] / torch.tensor([self.cfg.image_size_x, self.cfg.image_size_y], device=self.cfg.gpu_id)) * landmarks_gt[:, :, :, 2:3]) ** 2).sum(-1).sum(-2).mean()
                reg_loss = self.face_model.reg_loss(self.cfg.reg_id_weight, self.cfg.reg_exp_weight)
                loss = pro_loss + reg_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 更新进度条
                progress_bar.set_postfix(loss=loss.item())

                # if abs(loss.item() - prev_loss) < 1e-10 and abs(loss.item() - pprev_loss) < 1e-9:
                if abs(loss.item() - prev_loss) < 1e-3 and abs(loss.item() - pprev_loss) < 1e-2:
                    break
                else:
                    pprev_loss = prev_loss
                    prev_loss = loss.item()
        
        # 关闭进度条
        progress_bar.close()

        import ipdb
        # ipdb.set_trace()
        log = {
            'frames': frames[::len(self.cfg.camera_ids)],
            'landmarks_gt': landmarks_gt,
            'landmarks_2d': landmarks_2d.detach(),
            'face_model': self.face_model,
            'intrinsics': intrinsics0,
            'extrinsics': extrinsics0
        }
        self.recorder.log(log)


    def project(self, points_3d, intrinsic, extrinsic):
        points_3d = points_3d.permute(0,2,1)
        calibrations = torch.bmm(intrinsic, extrinsic)
        points_2d = self.camera.project(points_3d, calibrations)
        points_2d = points_2d.permute(0,2,1)
        return points_2d



class MyFitter_GA(MyFitter):
    def __init__(self, cfg, dataset, face_model, camera, recorder, device):
        self.cfg = cfg
        self.dataset = dataset
        self.face_model = face_model
        self.camera = camera
        self.recorder = recorder
        self.device = device

        self.optimizers = [torch.optim.Adam([{'params' : self.face_model.pose, 'lr' : 1e-2}]),
                           torch.optim.Adam([{'params' : self.face_model.parameters(), 'lr' : 1e-3}])]
    



