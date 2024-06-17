# import ipdb; ipdb.set_trace()

import os
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import os
import torch
import argparse

from config.config import config
from lib.LandmarkDataset import LandmarkDataset
from lib.Recorder import Recorder
from lib.Fitter import Fitter
from lib.face_models import get_face_model
from lib.Camera import Camera

# import os
# os.environ['PYGLET_WINDOW'] = 'dummy'

# import os
# os.environ["PYOPENGL_PLATFORM"] = "osmesa"


if __name__ == '__main__':
    import ipdb
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/MEAD_M003_single_vid.yaml')
    arg = parser.parse_args()

    cfg = config()
    cfg.load(arg.config)
    cfg = cfg.get_cfg()

    device = torch.device('cuda:%d' % cfg.gpu_id)
    torch.cuda.set_device(cfg.gpu_id)

    source_folder = cfg.data_source
    print('processing %s' % source_folder)

    emo_folders = sorted(os.listdir(source_folder))
    for emo_folder in emo_folders:
        print('processing %s' % emo_folder)
        level_folders = sorted(os.listdir(os.path.join(source_folder, emo_folder)))
        for level_folder in level_folders:
            print('processing %s' % level_folder)
            trial_folders = sorted(os.listdir(os.path.join(source_folder, emo_folder, level_folder)))
            for trial_folder in trial_folders:
                print('processing %s' % trial_folder)
                lmk_folder = os.path.join(source_folder, emo_folder, level_folder, trial_folder, 'landmarks')
                camera_folder = os.path.join(source_folder, emo_folder, level_folder, trial_folder, 'cameras')
                param_folder = os.path.join(source_folder, emo_folder, level_folder, trial_folder, 'params')
                dataset = LandmarkDataset(landmark_folder=lmk_folder, camera_folder=camera_folder)
                face_model = get_face_model(cfg.face_model, batch_size=len(dataset), device=device)
                camera = Camera(image_size=cfg.image_size)
                recorder = Recorder(save_folder=param_folder, camera=camera, visualize=cfg.visualize, save_vertices=cfg.save_vertices)

                fitter = Fitter(cfg, dataset, face_model, camera, recorder, device)
                fitter.run()




    # dataset = LandmarkDataset(landmark_folder=cfg.landmark_folder, camera_folder=cfg.camera_folder)
    # face_model = get_face_model(cfg.face_model, batch_size=len(dataset), device=device)
    # camera = Camera(image_size=cfg.image_size)
    # recorder = Recorder(save_folder=cfg.param_folder, camera=camera, visualize=cfg.visualize, save_vertices=cfg.save_vertices)

    # fitter = Fitter(cfg, dataset, face_model, camera, recorder, device)
    # # ipdb.set_trace()
    # fitter.run()
