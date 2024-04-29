import os
from yacs.config import CfgNode as CN
 


class config():

    def __init__(self):

        self.cfg = CN()
        self.cfg.image_folder = ''
        self.cfg.camera_folder = ''
        self.cfg.landmark_folder = ''
        self.cfg.param_folder = ''
        self.cfg.gpu_id = 0
        self.cfg.camera_ids = []
        self.cfg.image_size = 512
        self.cfg.face_model = 'BFM'
        self.cfg.reg_id_weight = 1e-6
        self.cfg.reg_exp_weight = 1e-6
        self.cfg.visualize = False
        self.cfg.save_vertices = False

        # NeRSemble_ga
        self.cfg.root_folder = ''
        self.cfg.image_suffix = 'images'
        self.cfg.landmark_suffix = 'landmarks'
        self.cfg.camera_suffix = 'cameras'
        self.cfg.param_suffix = 'params'
        self.cfg.image_size_x = 1920
        self.cfg.image_size_y = 1080

        # MEAD_single_vid
        self.cfg.data_source = ''

    def get_cfg(self):
        return  self.cfg.clone()
    
    def load(self, config_file):
         self.cfg.defrost()
         self.cfg.merge_from_file(config_file)
         self.cfg.freeze()
