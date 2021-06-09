from joblib import Parallel, delayed
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

max_n_neighbors = 10

k_range = np.arange(max_n_neighbors) + 1
X_train, y_train = np.load("../data/X_train.npy"), np.load("../data/y_train.npy")
X_val, y_val = np.load("../data/X_val.npy"), np.load("../data/y_val.npy")


def worker(k):
    model = KNeighborsClassifier(n_neighbors=k, weights="distance")
    model.fit(X_train, y_train)
    probas = model.predict_proba(X_val)
    with open("../out/knn_probas_pred_%d.npy" % k, "wb") as f:
        np.save(f, probas)


Parallel(n_jobs=2)(delayed(worker)(k) for k in k_range)
