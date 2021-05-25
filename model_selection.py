import numpy as np
import pickle
from sklearn.metrics import accuracy_score

jobs = np.arange(10)

y_val = np.load("iwildcam/y_val.npy")

C_range = np.logspace(-5, 5, 10)

best_C = None
best_acc = -1
best_id = -1

for job_id in jobs:
    if job_id != 2 and job_id != 3:
        y_pred = pickle.dump(open("out/lr_y_pred_%d.p" % job_id, "rb"))
        acc = accuracy_score(y_val, y_pred)

        if acc > best_acc:
            best_acc = acc
            best_C = C_range[job_id]
            best_id = job_id

pickle.dump(best_acc, open("metrics/best_lr_acc.p", "wb"))
pickle.dump(best_C, open("metrics/best_lr_C.p", "wb"))
pickle.dump(best_id, open("metrics/best_lr_id.p", "wb"))

best_lr_coef = pickle.load(open("out/lr_coef_%d" % best_id, "rb"))
best_lr_intercept = pickle.load(open("out/lr_intercept_%d" % best_id, "rb"))
pickle.dump(best_lr_coef, open("metrics/best_lr_coef.p", "wb"))
pickle.dump(best_lr_intercept, open("metrics/best_lr_intercept.p", "wb"))