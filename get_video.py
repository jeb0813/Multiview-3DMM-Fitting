import os
import cv2

import numpy as np

import ipdb

folder_img = '/data/chenziang/codes/Multiview-3DMM-Fitting/MEAD_MONO_single_vid/M003_25/angry/level_3/001/images'
folder_mask = '/data/chenziang/codes/Multiview-3DMM-Fitting/MEAD_MONO_single_vid/M003_25/angry/level_3/001/params'

folder_output = '/data/chenziang/codes/Multiview-3DMM-Fitting/temp_demo'

if __name__ == "__main__":
    frames = os.listdir(folder_img)
    frames.sort()

    # ipdb.set_trace()
    for frame in frames:
        path_img = os.path.join(folder_img, frame, 'image_2.jpg')
        path_mask = os.path.join(folder_mask, frame, 'vis_0.jpg')

        img = cv2.imread(path_img)
        mask = cv2.imread(path_mask)

        result = np.hstack((img, mask))
        cv2.imwrite(os.path.join(folder_output, frame + '.jpg'), result)


