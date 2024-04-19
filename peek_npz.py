import os
import numpy as np


ga_params = "/data/chenziang/codes/GaussianAvatars/data/074/cluster/ikarus/sqian/project/dynamic-head-avatars/code/multi-view-head-tracker/export/074_EMO-1_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine/flame_param/00000.npz"

# 这个是BFM的参数，不适用
nersemble_params = "/data/chenziang/codes/Gaussian-Head-Avatar/NeRSemble/074/params/0000/params.npz"

ga = np.load(ga_params)
nersemble = np.load(nersemble_params)

# 查看字段
print(ga.files)
print(nersemble.files)

