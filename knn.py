import sys
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import pickle

# Get Slurm array job ID.
# job_id = 1 # test
job_id = int(sys.argv[1])
k_range = np.arange(10) # length needs to be same as number of jobs.

X_train, y_train = np.load("iwildcam/Z_train.npy"), np.load("iwildcam/y_train.npy")

model = KNeighborsClassifier(n_neighbors=k_range[job_id], weights='distance')

model.fit(X_train, y_train)

X_val, y_val = np.load("iwildcam/Z_val.npy"), np.load("iwildcam/y_val.npy")
y_pred = model.predict(X_val)

result = confusion_matrix(y_val, y_pred, labels=model.classes_)

pickle.dump(result, open("out/knn_result_%d.p" % job_id, "wb"))