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

def softmax(Z):
    np.clip(Z, -10, 10)
    norm_constant = np.exp(Z).sum(axis=1)
    
    return np.exp(Z) / norm_constant

def predict_proba(model, x):
    """ Gives a prediction of x
            :param ``numpy.array`` x: input whose label is to predict
            :return:  value of the prediction
    """
    self = model
    formatted_x = np.ones((x.shape[0], self.n_features + self.fit_intercept))
    formatted_x[:, self.fit_intercept:] = x
    casted_sol = np.reshape(self.solution, (self.n_features + self.fit_intercept, self.n_classes))
    probas = softmax(np.dot(formatted_x, casted_sol))
    # predictions = np.argmax(probas, axis=1)
    return probas
        






