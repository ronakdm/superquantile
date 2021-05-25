import numpy as np
import pickle
from utils import softmax
from sklearn.metrics import accuracy_score, precision_score, recall_score

# To compute:
# - Test accuracy.
# - p-superquantile loss
# - p-superquantile precision across all classes.
# - p-superquantile recall across all classes.
# - p-superquantile accuracy across all locations

def sq(arr, p):
    """
    p-superquantile of numpy array.
    """
    np.mean(arr[arr >= np.quantile(arr, p)])

def compute_metrics(probas, y_true, metadata, p=0.8):
    # 1. Test accuracy.
    y_pred = np.argmax(probas, axis=1)
    acc = accuracy_score(y_true, y_pred)

    # 2. p-superquantile log loss.
    losses = -np.log(probas)[:, y_true]
    sq_loss = sq(losses, p)

    # 3. p-superquantile precision across all classes.
    sq_prec = sq(precision_score(y_true, y_pred), p)

    # 4. p-superquantile recall across all classes.
    sq_rec = sq(recall_score(y_true, y_pred), p)

    # 5. p-superquantile accuracy across all locations
    locations = np.unique(metadata)
    loc_accs = []
    for loc in locations:
        loc_acc = accuracy_score(y_true[locations==loc], y_pred[locations==loc])
        loc_accs.append(loc_acc)
    sq_loc = sq(loc_accs, p)

    metrics = {
        "accuracy" : acc,
        "%0.2f-spq loss" : sq_loss,
        "%0.2f-spq precision" : sq_prec,
        "%0.2f-spq recall" : sq_rec,
        "%0.2f-spq location accuracy" : sq_loc,
    }

    print(metrics)

    return metrics

best_lr_id = pickle.load(open("metrics/best_lr_id.p", "rb"))
coef = pickle.load(open("metrics/lr_coef_%d.p" % best_lr_id, "rb"))
intercept = pickle.load(open("metrics/lr_intercept_%d.p" % best_lr_id, "rb"))

X_test = np.load("iwildcam/Z_test")
y_test = np.load("iwildcam/y_test")
metadata = np.load("out/test_meta_0.npy")

Z = np.dot(X_test, coef) + intercept.reshape(-1, 1)
lr_probas = softmax(Z)

lr_metrics = compute_metrics(lr_probas, y_test, metadata, p=0.8)



