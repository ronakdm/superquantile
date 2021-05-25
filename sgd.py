
import sys
import numpy as np
from sklearn.linear_model import SGDClassifier
# from sklearn.metrics import confusion_matrix
import pickle

# Get Slurm array job ID.

job_id = int(sys.argv[1]) - 1
C_range = np.logspace(-5, 5, 10) # length needs to be same as number of jobs.

X_train, y_train = np.load("iwildcam/Z_train.npy"), np.load("iwildcam/y_train.npy")

model = SGDClassifier(loss='log', C=C_range[job_id], solver='saga', max_iter=1000)

model.fit(X_train, y_train)

pickle.dump(model.coef_, open("out/lr_coef_%d.p" % job_id, "wb"))
pickle.dump(model.intercept_, open("out/lr_intercept_%d.p" % job_id, "wb"))

X_val, y_val = np.load("iwildcam/Z_val.npy"), np.load("iwildcam/y_val.npy")
y_pred = model.predict(X_val)

pickle.dump(y_pred, open("out/lr_y_pred_%d.p" % job_id, "wb"))