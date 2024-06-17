import sys
PATH = '/data/chenziang/codes/Multiview-3DMM-Fitting/submodules/wavlm'
sys.path.append(PATH)

import os
import numpy as np
import argparse
import torch
import ipdb

from WavLM import WavLM, WavLMConfig
from scipy.io import wavfile

device = torch.device('cuda:0')
torch.cuda.set_device(0)

# load the pre-trained checkpoints
checkpoint = torch.load('/data/chenziang/codes/Multiview-3DMM-Fitting/submodules/wavlm/checkpoints/WavLM-Base+.pt')
cfg = WavLMConfig(checkpoint['cfg'])
model = WavLM(cfg).to(device)
model.load_state_dict(checkpoint['model'])
model.eval()


def extract_feature(path):
    # ipdb.set_trace()
    sample_rate, audio_data = wavfile.read(path)
    audio_np = audio_data.astype(np.float32)
    audio_np = np.expand_dims(audio_np, axis=0)

    audio_tensor = torch.from_numpy(audio_np).to(device)
    rep = model.extract_features(audio_tensor)[0]

    return rep

def split_to_frame(npy_path, out_folder):
    audio_path = os.path.join(out_folder, 'audio')
    os.makedirs(audio_path, exist_ok=True)

    audio_feature = np.load(npy_path)[0]
    frames = sorted(os.listdir(os.path.join(out_folder, 'images')))

    # assert len(frames)*2 <= audio_feature.shape[0]

    for i,frame in enumerate(frames):
        # ipdb.set_trace()
        audio_frame_path = os.path.join(audio_path, frame)
        os.makedirs(audio_frame_path, exist_ok=True)
        if i*2+1 >= audio_feature.shape[0]:
            audio_feature_frame = audio_feature[-1]
        else:
            audio_feature_frame = audio_feature[i*2:i*2+2]
            audio_feature_frame = np.mean(audio_feature_frame, axis=0)

        np.save(os.path.join(audio_frame_path,'audio.npy'), audio_feature_frame)






def extract_audio(id,emos=['neutral']):
    # ipdb.set_trace()

    # 遍历emo
    for emo in emos:
        print("Processing %s %s" % (id, emo))
        emo_path = os.path.join(DATA_SOURCE, id, 'audio', emo)

        audio_folders_level = os.listdir(emo_path)
        audio_folders_level.sort()

        # 遍历level
        for audio_folder_level in audio_folders_level:
            out_folder_level = os.path.join(DATA_OUTPUT, id, emo, audio_folder_level)

            level_path = os.path.join(emo_path, audio_folder_level)
            print("Processing %s %s %s" % (id, emo, audio_folder_level))
            audio_folders = os.listdir(level_path)
            audio_folders.sort()

            # ipdb.set_trace()

            for audio in audio_folders:
                # ipdb.set_trace() 
                if not audio.endswith('.wav'):
                    continue

                print("Processing %s" % (audio))

                out_folder = os.path.join(out_folder_level,audio[:3])

                # 将音频复制过去
                out_audio_path = os.path.join(out_folder, 'audio.wav')
                os.system('cp %s %s' % (os.path.join(level_path, audio), out_audio_path))

                save_path = os.path.join(out_folder, 'audio.npy')

                if not os.path.exists(save_path):
                    ret = extract_feature(out_audio_path)
                    np.save(save_path, ret.cpu().detach().numpy())

                # ipdb.set_trace()
                split_to_frame(save_path, out_folder)



def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_source', type=str, default='/data/chenziang/codes/Multiview-3DMM-Fitting/MEAD')
    parser.add_argument('--data_output', type=str, default='../MEAD_MONO_single_vid')
    parser.add_argument('--left_up', type=list, default=[(1920-1080)//2, 0])
    parser.add_argument('--crop_size', type=list, default=[1080, 1080])
    parser.add_argument('--size', type=list, default=[1080, 1080])
    parser.add_argument('--size_lowres', type=list, default=[256, 256])
    parser.add_argument('--subject_id', type=str, default='M003_25')
    return parser.parse_args()



if __name__ == "__main__":
    """
    单目数据
    """
    args = get_args()

    LEFT_UP = args.left_up
    CROP_SIZE = args.crop_size
    SIZE = args.size
    SIZE_LOWRES = args.size_lowres
    DATA_SOURCE = args.data_source
    DATA_OUTPUT = args.data_output
    subject_id = args.subject_id


    if not os.path.exists(DATA_OUTPUT):
        os.makedirs(DATA_OUTPUT, exist_ok=True)

    import ipdb
    emos = ['neutral', 'angry', 'contempt', 'disgusted', 'fear', 'happy', 'sad', 'surprised']
    # emos = ['neutral']
    views = ['front']
    # ipdb.set_trace() 
    extract_audio(subject_id, emos)

