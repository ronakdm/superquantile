import numpy as np
import pickle
from utils import resnet18
from sklearn import metrics

print("\nPrinting feature matrices statistics...")
print("---------------------------------------")

X_val = resnet18(split='val', verbose=False)[0]
Z_val = np.load("iwildcam/Z_val.npy")

_, d = X_val.shape
_, k = Z_val.shape

print("Original number of features (ResNet18):", d)
print("Reduced number of features:", k)

for split in ["train", "val", "test"]:
    X = np.load("iwildcam/Z_%s.npy" % split)
    print("Embedded X %s shape:\t" % split, X.shape)

print("\nPrinting label statistics...")
print("---------------------------------------")

y_train = np.load("iwildcam/y_train.npy")
train_classes = np.unique(y_train)
print("Num. labels in y train:\t", len(train_classes))

y_val = np.load("iwildcam/y_val.npy")
val_classes = np.unique(y_val)
print("Num. labels in y val:\t", len(val_classes))

y_test = np.load("iwildcam/y_test.npy")
test_classes = np.unique(y_test)
print("Num. labels in y test:\t", len(test_classes))

print("Num. classes in both y train and y val:\t", len(np.intersect1d(train_classes, val_classes)))
print("Num. classes in both y train and y test:\t", len(np.intersect1d(train_classes, test_classes)))

jobs = {
    "knn" : [1, 2, 3, 4, 5, 6, 7, 8, 9],
    "lr" : [0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13 ,14]
}
for model in ["lr", "knn"]:

    print("\nPrinting %s prediction statistics..." % model)
    print("---------------------------------------")

    for i, job in enumerate(jobs[model]):
        print("Job %d:" % i)
        y_pred = pickle.load(open("out/%s_y_pred_%d.p" % (model, job), "rb"))
        pred_classes = np.unique(y_pred)
        confusion_matrix = metrics.confusion_matrix(y_pred, y_val)

        print("\t Num. predicted classes: %d/%d" % (len(pred_classes), len(val_classes)))
        print("\t Confusion matrix shape:", confusion_matrix.shape)
        print("\t Num. classes in either y pred or y val:", len(np.union1d(pred_classes, val_classes)))

