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
1. preprocess_mead.py
2. remove_background_mead.py
3. detect_landmarks_mead.py
4. fitting_mead.py