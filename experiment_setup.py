import numpy as np
import pickle

def resnet18(split="val", verbose=False):
    X_features = np.load('iwildcam/resnet18_features/%s_features.npy' % split)
    X_metadata = np.load('iwildcam/resnet18_features/%s_metadata.npy' % split)

    X = np.concatenate((X_features, X_metadata), axis=1)
    y = np.load('iwildcam/resnet18_features/%s_labels.npy' % split)

    if verbose:
        print("Loading ResNet18 '%s' data..." % split)
        print("X shape (features):", X_features.shape)
        print("X shape (metadata):", X_metadata.shape)
        print("X shape:", X.shape)
        print("y shape:", y.shape)

    return X, y

def C_hyperparameter():
    return np.logspace(-5, 5, 100)





