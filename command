source activate gha

CUDA_VISIBLE_DEVICES=3 python detect_landmarks.py --config config/NeRSemble_074.yaml

CUDA_VISIBLE_DEVICES=3 python fitting.py --config config/NeRSemble_074.yaml

CUDA_VISIBLE_DEVICES=3 python detect_landmarks.py --config config/MEAD_003.yaml

CUDA_VISIBLE_DEVICES=3 python fitting.py --config config/MEAD_003.yaml


1. preprocess_nersemble_ga.py
2. remove_background_nersemble_ga.py
3. detect_landmarks_ga.py
4. fitting


CUDA_VISIBLE_DEVICES=3 python fitting_ga.py --config config/NeRSemble_ga_074_format.yaml
