source activate gha

CUDA_VISIBLE_DEVICES=3 python detect_landmarks.py --config config/NeRSemble_074.yaml

CUDA_VISIBLE_DEVICES=3 python fitting.py --config config/NeRSemble_074.yaml

CUDA_VISIBLE_DEVICES=3 python detect_landmarks.py --config config/MEAD_003.yaml

CUDA_VISIBLE_DEVICES=2 python fitting.py --config config/MEAD_003.yaml

0. colmap2json
1. preprocess_nersemble_ga.py
2. remove_background_nersemble_ga.py
3. detect_landmarks_ga.py
4. fitting


CUDA_VISIBLE_DEVICES=3 python fitting_ga.py --config config/NeRSemble_ga_074_format.yaml



我们假设json文件是已经预处理好了的
1. preprocess
CUDA_VISIBLE_DEVICES=2 python preprocess_mead_single_vid.py
2. remove_background
CUDA_VISIBLE_DEVICES=2 python remove_background_mead_single_vid.py
3. detect_landmarks
CUDA_VISIBLE_DEVICES=2 python detect_landmarks_mead_single_vid.py --config config/MEAD_M003_single_vid.yaml
  1结束后, 2可以和3并行
4. fitting_mead
CUDA_VISIBLE_DEVICES=2 python fitting_mead_single_vid.py --config config/MEAD_M003_single_vid.yaml
5. 划分数据集
utils/split.py
6. face parsing 
首先生成masked image
CUDA_VISIBLE_DEVICES=3 python face_parsing.py --config config/MEAD_M003_single_vid.yaml


