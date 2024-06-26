# 3DMM Fitting for Multiview (monocular) Videos
This is a very fast offline fitting framework, which uses only landmarks. Currently commonly used 3DMM models: BFM, FaceVerse and FLAME are supported. 
<img src="gifs/NeRSemble_031_bfm.gif" alt="demo" width="840" height="460"/> 
<img src="gifs/sample_video.gif" alt="demo" width="840" height="210"/> 
More results are in "gifs/".

## Installation
### Requirements
* Create a conda environment.
```
conda env create -f environment.yaml
```
* Install [Pytorch3d](https://github.com/facebookresearch/pytorch3d).
```
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1120/download.html
```

### BFM
* Follow the installation of [3DMM-Fitting-Pytorch](https://github.com/ascust/3DMM-Fitting-Pytorch) to generate "BFM09_model_info.mat" and put it into "assets/BFM/".

### FaceVerse
* Download [FaceVerse version 2](https://github.com/LizhenWangT/FaceVerse) and put "faceverse_simple_v2.npy" into "assets/FaceVerse/".

### FLAME
* Download [FLAME model](https://flame.is.tue.mpg.de/) and put "generic_model.pkl" into "assets/FLAME/".
* Download the embeddings from [RingNet](https://github.com/soubhiksanyal/RingNet/tree/master/flame_model) and put "flame_dynamic_embedding.npy" and "flame_static_embedding.pkl" into "assets/FLAME/".

### Datasets
Make the dataset according to the following directory structure:
```
data_root
│   └── images
│   │   └── {frame_0}
|   |   |   └── image_{camera_id_0}.jpg
|   |   |   └── image_{camera_id_1}.jpg
|   |   |   └── ...
│   │   └── {frame_1}
|   |   |   └── image_{camera_id_0}.jpg
|   |   |   └── image_{camera_id_1}.jpg
|   |   |   └── ...
|   |   └── ...
│   └── cameras
│   │   └── {frame_0}
|   |   |   └── camera_{camera_id_0}.npz
|   |   |   └── camera_{camera_id_1}.npz
|   |   |   └── ...
│   │   └── {frame_1}
|   |   |   └── camera_{camera_id_0}.npz
|   |   |   └── camera_{camera_id_1}.npz
|   |   |   └── ...
|   |   └── ...
```
I provide a script "preprocess/preprocess_monocular_video.py" for preprocessing [NeRSemble dataset](https://tobias-kirschstein.github.io/nersemble/) to such a structured dataset. Please apply to download it and put it into "path/to/raw_NeRSemble/".
```
cd preprocess
python preprocess_nersemble.py
```
Besides, I also provide a script "preprocess/preprocess_monocular_video.py" for converting a monocular video. 
```
cd preprocess
python preprocess_monocular_video.py
```

## Multiview (monocular) Fitting
First, edit the config file, for example "config/NeRSemble_031.yaml".
Second, detect 2D landmarks for all the input images.
```
python detect_landmarks.py --config config/NeRSemble_031.yaml
```
Third, fit 3DMM model.
```
python fitting.py --config config/NeRSemble_031.yaml
```


## Acknowledgement
Part of the code is borrowed from [FLAME_PyTorch](https://github.com/soubhiksanyal/FLAME_PyTorch).
