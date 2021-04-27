import numpy as np
import pickle

def resnet18(split="val", verbose=False):
    X = np.load('iwildcam/resnet18_features/%s_features.npy' % split)
    y = np.load('iwildcam/resnet18_features/%s_labels.npy' % split)

    if verbose:
        print("Loading ResNet18 '%s' data..." % split)
        print("X shape:", X.shape)
        print("y shape:", y.shape)

    return X, y

def C_hyperparameter():
    return np.logspace(-5, 5, 30)





