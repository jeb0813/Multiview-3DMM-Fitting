import os
import torch
import argparse

from config.config import config
from lib.LandmarkDataset import LandmarkDataset, LandmarkDatasetGA
from lib.Recorder import Recorder
from lib.Fitter import Fitter, MyFitter, MyFitter_GA
from lib.face_models import get_face_model
from lib.Camera import Camera, CameraGA

import os
os.environ['PYGLET_WINDOW'] = 'dummy'


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

    trials = sorted(os.listdir(cfg.root_folder))
    for trial in trials:
        # ipdb.set_trace()
        if not(trial.startswith('EMO') or trial.startswith('EXP')):
            continue
        print("Processing trial: ", trial)
        landmark_folder = os.path.join(cfg.root_folder, trial, cfg.landmark_suffix)
        camera_folder = os.path.join(cfg.root_folder, trial, cfg.camera_suffix)

        camera_ids = ['{:02d}'.format(i) for i in range(len(cfg.camera_ids))]
        dataset = LandmarkDatasetGA(landmark_folder=landmark_folder, camera_folder=camera_folder, camera_ids=camera_ids)
        # ipdb.set_trace()
        face_model = get_face_model(cfg.face_model, batch_size=len(dataset), device=device)
        camera = CameraGA(image_size_x=cfg.image_size_x, image_size_y=cfg.image_size_y)
        param_folder = os.path.join(cfg.root_folder, trial, cfg.param_suffix)
        recorder = Recorder(save_folder=param_folder, camera=camera, visualize=cfg.visualize, save_vertices=cfg.save_vertices)

        fitter = MyFitter_GA(cfg, dataset, face_model, camera, recorder, device)
        # ipdb.set_trace()
        fitter.run()

