
import os
import sys
import numpy as np
import pickle

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from spqr.spqr import DRLogisticRegression

# Get Slurm array job ID.
job_id = int(sys.argv[1]) - 1

# TODO: Set hyperparameters.
p = 0.9
mu = 1.0
lmbda = 1

def sigmoid(Z):
    np.clip(Z, -10, 10)
    return 1 / (1 + np.exp(-Z))

def predict_proba(model, x):
    """ Gives a prediction of x
            :param ``numpy.array`` x: input whose label is to predict
            :return:  value of the prediction
    """
    self = model
    formatted_x = np.ones((x.shape[0], self.n_features + self.fit_intercept))
    formatted_x[:, self.fit_intercept:] = x
    casted_sol = np.reshape(self.solution, (self.n_features + self.fit_intercept, self.n_classes))
    probas = sigmoid(np.dot(formatted_x, casted_sol))
    # predictions = np.argmax(probas, axis=1)
    return probas

X_train, y_train = np.load("iwildcam/Z_train.npy"), np.load("iwildcam/y_train.npy")

model = DRLogisticRegression(p=p, mu=mu, lmbda=lmbda)

model.fit(X_train, y_train)

X_val, y_val = np.load("iwildcam/Z_val.npy"), np.load("iwildcam/y_val.npy")

probas = predict_proba(model, X_val)
y_pred = model.predict(X_val)

pickle.dump(probas, open("out/drlr_probas_%d.p" % job_id, "wb"))
pickle.dump(y_pred, open("out/drlr_y_pred_%d.p" % job_id, "wb"))