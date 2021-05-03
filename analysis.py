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
    precision[i] = diagonal / row_sums
    recall[i] = diagonal / column_sums
    f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])

# Superquantile accruracy