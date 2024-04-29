import os 
import json

num_val = 5

train = dict()
val = dict()

source_folder = '/data/chenziang/codes/Multiview-3DMM-Fitting/MEAD_MONO_single_vid/M003'
emo_folders = sorted(os.listdir(source_folder))

for emo_folder in emo_folders:
    train_emo = dict()
    val_emo = dict()

    level_folders = sorted(os.listdir(os.path.join(source_folder, emo_folder)))
    for level_folder in level_folders:
        trial_folders = sorted(os.listdir(os.path.join(source_folder, emo_folder, level_folder)))

        train_emo[level_folder] = trial_folders[:-num_val]
        val_emo[level_folder] = trial_folders[-num_val:]


    train[emo_folder] = train_emo
    val[emo_folder] = val_emo

json_path_train = os.path.join(source_folder, 'train.json')
json_path_val = os.path.join(source_folder, 'val.json')

with open(json_path_train, 'w') as f:
    json.dump(train, f, indent=4)

with open(json_path_val, 'w') as f:
    json.dump(val, f, indent=4)




