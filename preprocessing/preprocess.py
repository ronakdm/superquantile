import numpy as np
from utils import resnet18

for split in ["train", "val", "test"]:
    y = resnet18(split=split, verbose=True)[1]
    with open("iwildcam/y_%s.npy" % split, "wb") as f:
        np.save(f, y)