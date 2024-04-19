import os
import torch
import tqdm
import glob
import numpy as np
import cv2
import face_alignment
import argparse

from config.config import config


def run_detect_trial(trial):

    source_folder = os.path.join(cfg.root_folder, trial, cfg.image_suffix)
    output_folder = os.path.join(cfg.root_folder, trial, cfg.landmark_suffix)
    os.makedirs(output_folder, exist_ok=True)

    img_paths = sorted(os.listdir(source_folder))
    img_paths = [os.path.join(source_folder, img_path) for img_path in img_paths]
    batch_size = 16
    # ipdb.set_trace()
    batches = [img_paths[i:i+batch_size] for i in range(0, len(img_paths), batch_size)]

    for img_paths in tqdm.tqdm(batches):
        images = np.stack([cv2.imread(image_path)[:, :, ::-1] for image_path in img_paths])
        images = torch.from_numpy(images).float().permute(0, 3, 1, 2).to(device)

        results = fa.get_landmarks_from_batch(images, return_landmark_score=True)
        for i in range(len(results[0])):
            if results[1][i] is None:
                results[0][i] = np.zeros([68, 3], dtype=np.float32)
                results[1][i] = [np.zeros([68], dtype=np.float32)]
            if len(results[1][i]) > 1:
                total_score = 0.0
                for j in range(len(results[1][i])):
                    if np.sum(results[1][i][j]) > total_score:
                        total_score = np.sum(results[1][i][j])
                        landmarks_i = results[0][i][j*68:(j+1)*68]
                        scores_i = results[1][i][j:j+1]
                results[0][i] = landmarks_i
                results[1][i] = scores_i
                
        landmarks = np.concatenate([np.stack(results[0])[:, :, :2], np.stack(results[1]).transpose(0, 2, 1)], -1)
        for i, image_path in enumerate(img_paths):
            landmarks_path = os.path.join(output_folder, image_path.split('/')[-1].replace('.png', '.npy'))
            np.save(landmarks_path, landmarks[i])


def exception_handle(trial):
    landmark_folder = os.path.join(cfg.root_folder, trial, cfg.landmark_suffix)
    landmark_paths = sorted(os.listdir(landmark_folder))
    for landmark_path in landmark_paths:
        landmarks = np.load(os.path.join(landmark_folder, landmark_path))
        # 如果是全0的landmark
        if np.sum(landmarks[:, :2]) == 0:
            print("Warning: ", os.path.join(landmark_folder, landmark_path))
            # 用最近的一个landmark替代
            idx=int(landmark_path[:5])
            cam_id = landmark_path[6:8]
            shift = 1
            while True:
                if idx - shift >= 0:
                    new_landmark_path = os.path.join(landmark_folder, '%05d_%s.npy' % (idx-shift, cam_id))
                    if os.path.exists(new_landmark_path):
                        new_landmarks = np.load(new_landmark_path)
                        if np.sum(new_landmarks[:, :2]) != 0:
                            # 删除全0的landmark
                            os.remove(os.path.join(landmark_folder, landmark_path))
                            # 复制重命名
                            os.system('cp %s %s' % (new_landmark_path, os.path.join(landmark_folder, landmark_path)))
                            break
                else:
                    new_landmark_path = os.path.join(landmark_folder, '%05d_%s.npy' % (idx+shift, cam_id))
                    if os.path.exists(new_landmark_path):
                        new_landmarks = np.load(new_landmark_path)
                        if np.sum(new_landmarks[:, :2]) != 0:
                            # 删除全0的landmark
                            os.remove(os.path.join(landmark_folder, landmark_path))
                            # 复制重命名
                            os.system('cp %s %s' % (new_landmark_path, os.path.join(landmark_folder, landmark_path)))
                            break
                shift += 1
            
            print("Replace with: ", new_landmark_path)

if __name__ == '__main__':
    import ipdb

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/NeRSemble_ga_074.yaml')
    arg = parser.parse_args()

    cfg = config()
    cfg.load(arg.config)
    cfg = cfg.get_cfg()

    device = torch.device('cuda:%d' % cfg.gpu_id)
    torch.cuda.set_device(cfg.gpu_id)

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, flip_input=False, face_detector='blazeface', device='cuda:%d' % cfg.gpu_id)

    trials = sorted(os.listdir(cfg.root_folder))
    for trial in trials:
        if not(trial.startswith('EMO') or trial.startswith('EXP')):
            continue
        print("Processing trial: ", trial)
        run_detect_trial(trial)

        # 异常处理
        exception_handle(trial)
