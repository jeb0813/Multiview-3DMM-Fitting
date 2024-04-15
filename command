source activate gha

CUDA_VISIBLE_DEVICES=3 python detect_landmarks.py --config config/NeRSemble_074.yaml

CUDA_VISIBLE_DEVICES=3 python fitting.py --config config/NeRSemble_074.yaml

CUDA_VISIBLE_DEVICES=3 python detect_landmarks.py --config config/MEAD_003.yaml

CUDA_VISIBLE_DEVICES=3 python fitting.py --config config/MEAD_003.yaml