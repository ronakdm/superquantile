import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn import metrics

# K-Nearest Neighbors

jobs = {
    "knn" : np.arange(1, 10),
    "lr" : np.arange(15)
}
y_val = np.load("iwildcam/y_val.npy")
n_classes = 182 # Hard coded for now, should be saved from model.

for model in ["lr", "knn"]:

    # Accuracy and Macro F1

    n_jobs = len(jobs[model])

    accuracy = np.zeros(n_jobs)
    precision = np.zeros((n_jobs, n_classes))
    recall = np.zeros((n_jobs, n_classes))
    f1 = np.zeros((n_jobs, n_classes))

    for i, job in enumerate(jobs[model]):
        y_pred = pickle.load(open("out/%s_y_pred_%d.p" % (model, job), "rb"))
        confusion_matrix = metrics.confusion_matrix(y_pred, y_val)
        print(confusion_matrix.shape)
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

    with open("%s_metrics/accuracy.npy" % model, "wb") as f:
        np.save(f, accuracy)
    with open("%s_metrics/precision.npy" % model, "wb") as f:
        np.save(f, precision)
    with open("%s_metrics/recall.npy" % model, "wb") as f:
        np.save(f, recall)
    with open("%s_metrics/f1.npy" % model, "wb") as f:
        np.save(f, f1)
    with open("%s_metrics/macro_f1.npy" % model, "wb") as f:
        np.save(f, macro_f1)

