from joblib import Parallel, delayed
import numpy as np
from sklearn.linear_model import LogisticRegression

num_C_values = 10  # regularization constants.

C_range = np.logspace(-5, 5, num_C_values)
X_train, y_train = np.load("../data/X_train.npy"), np.load("../data/y_train.npy")
X_val, y_val = np.load("../data/X_val.npy"), np.load("../data/y_val.npy")


def worker(i, C):
    model = model = LogisticRegression(C=C, solver="liblinear", max_iter=3000)
    model.fit(X_train, y_train)
    probas = model.predict_proba(X_val)
    with open("../out/lr_probas_pred_%d.npy" % i, "wb") as f:
        np.save(f, probas)
    with open("../out/lr_coef_%d.npy" % i, "wb") as f:
        np.save(f, model.coef_)
    with open("../out/lr_intercept_%d.npy" % i, "wb") as f:
        np.save(f, model.intercept_)


Parallel(n_jobs=2)(delayed(worker)(i, C) for i, C in enumerate(C_range))
