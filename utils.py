import numpy as np

def resnet18(split="val", verbose=False):
    X = np.load('iwildcam/resnet18_features/%s_features.npy' % split)
    y = np.load('iwildcam/resnet18_features/%s_labels.npy' % split)

    if verbose:
        print("Loading ResNet18 '%s' data..." % split)
        print("X shape:", X.shape)
        print("y shape:", y.shape)

    return X, y

def compute_confusion_matrix(y_pred):
    y_train = np.load("out/y_train.npy")
    y_val = np.load("out/y_val.npy")

    total_classes = len(np.unique(y_train))

    mat = np.zeros((total_classes, total_classes))

    for i in range(len(y_val)):
        mat[y_val[i], y_pred[i]] += 1

    return mat
        






