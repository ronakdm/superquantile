import numpy as np
import matplotlib.pyplot as plt
import pickle

# K-Nearest Neighbors

n_jobs = 9
y_val = np.load("iwildcam/y_val.npy")
n_classes = 182 # Hard coded for now, should be saved from model.

# Accuracy and Macro F1

accuracy = np.zeros(n_jobs)
precision = np.zeros((n_jobs, n_classes))
recall = np.zeros((n_jobs, n_classes))
f1 = np.zeros((n_jobs, n_classes))

for i in range(n_jobs):
    confusion_matrix = pickle.load(open("out/knn_result_%d.p" % (i + 1), "rb"))
    accuracy[i] = np.trace(confusion_matrix) / np.sum(confusion_matrix)

    diagonal = np.diag(confusion_matrix)
    row_sums = np.sum(confusion_matrix, axis=0)
    column_sums = np.sum(confusion_matrix, axis=1)

    for j in range(n_classes):
        precision[i][j] = diagonal[j]/row_sums[j] if row_sums[j] > 0 else -1
        recall[i][j] = diagonal[j]/column_sums[j] if column_sums[j] > 0 else -1
        if precision[i][j] < 0 or recall[i][j] < 0 or precision[i][j] + recall[i][j] == 0:
            f1[i][j] = -1
        else:
            f1[i][j] = 2 * precision[i][j] * recall[i][j] / (precision[i][j] + recall[i][j])

macro_f1 = np.zeros(n_jobs)
for i in range(n_jobs):
    if np.all(f1[i] >= 0):
        macro_f1[i] = np.mean(f1[i])
    else:
        macro_f1[i] = -1

with open("knn_metrics/accuracy.npy", "wb") as f:
    np.save(f, accuracy)
with open("knn_metrics/precision.npy", "wb") as f:
    np.save(f, precision)
with open("knn_metrics/recall.npy", "wb") as f:
    np.save(f, recall)
with open("knn_metrics/f1.npy", "wb") as f:
    np.save(f, f1)
with open("knn_metrics/macro_f1.npy", "wb") as f:
    np.save(f, macro_f1)


# Superquantile accruracy