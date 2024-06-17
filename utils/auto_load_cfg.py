import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_id', type=str, default='M005')
    return parser.parse_args()

def write_cfg(subject_id,cfg_name):

    cfg = f"""
data_source: '/data/chenziang/codes/Multiview-3DMM-Fitting/MEAD_MONO_single_vid/{subject_id}'
gpu_id: 0
camera_ids: ['2']
image_size: 1080
face_model: 'BFM'
reg_id_weight: 1e-7
reg_exp_weight: 1e-7
save_vertices: True
visualize: False
    """

    with open(cfg_name, 'w') as f:
        f.write(cfg)

if __name__ == "__main__":
    args = get_args()
    subject_id = args.subject_id

    write_cfg(subject_id,f"/data/chenziang/codes/Multiview-3DMM-Fitting/config/MEAD_{subject_id}_single_vid.yaml")