# CUDA_VISIBLE_DEVICES should be defined when running this script

pwd=$(pwd)
process_path=$pwd'/preprocess'

sid=$1

# cd $process_path

# echo "Step 1"
# python preprocess_mead_single_vid.py --subject_id $sid 

# echo "Step 2"
# python remove_background_mead_single_vid.py --subject_id $sid

cd $pwd

# echo "Step 3"
# python utils/auto_load_cfg.py --subject_id $sid
# python detect_landmarks_mead_single_vid.py --config "config/MEAD_${sid}_single_vid.yaml"

echo "Step 4"
python fitting_mead_single_vid.py --config "config/MEAD_${sid}_single_vid.yaml"

echo "Step 5"
python utils/split.py --subject_id $sid

echo "Step 6"
python face_parsing.py --config config/MEAD_${sid}_single_vid.yaml
