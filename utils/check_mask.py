import os
import numpy as np
import cv2


# import ipdb
# ipdb.set_trace()


images_path = '/data/chenziang/codes/Multiview-3DMM-Fitting/MEAD_MONO/M003/neutral/level_1/images'
out_path = '/data/chenziang/codes/Multiview-3DMM-Fitting/temp'
timesteps = os.listdir(images_path)



for timestep in timesteps:
    timestep_path = os.path.join(images_path, timestep)
    image_path = os.path.join(timestep_path, 'image_2.jpg')
    mask_path = os.path.join(timestep_path, 'mask_2.jpg')
    image = cv2.imread(image_path).astype(np.float32)
    image/=255
    mask = cv2.imread(mask_path).astype(np.float32)
    mask /= 255

    output = image * mask
    output *= 255
    # rgb2bgr
    # output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(out_path, 'output_{}.jpg'.format(timestep)), output.astype(np.uint8))