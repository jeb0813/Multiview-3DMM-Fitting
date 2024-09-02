import torch

from scipy.io import wavfile
import soundfile as sf
from transformers import Wav2Vec2Model, Wav2Vec2Processor

# checkpoint_path = '/data/chenziang/codes/Multiview-3DMM-Fitting/checkpoints_wav2vec/wav2vec_big_960h.pt'
# checkpoint_path = '/data/chenziang/codes/Multiview-3DMM-Fitting/checkpoints_wav2vec/wav2vec_small_960h.pt'
checkpoint_path = '/data/chenziang/codes/Multiview-3DMM-Fitting/checkpoints_wav2vec/'
processor = Wav2Vec2Processor.from_pretrained(checkpoint_path)
model = Wav2Vec2Model.from_pretrained(checkpoint_path)    # 用于提取通用特征，768维



if __name__=="__main__":
    path = '/data/chenziang/codes/Multiview-3DMM-Fitting/MEAD_MONO_single_vid/M003_25/angry/level_1/001/audio.wav'
    # sample_rate, audio_data = wavfile.read(path)
    audio_data, sample_rate = sf.read(path)
    import ipdb; ipdb.set_trace()

    input_values = processor(audio_data, sampling_rate=sample_rate, return_tensors="pt").input_values
    wav2vec2 = model(input_values)['last_hidden_state'] 

