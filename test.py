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

def compute_metrics(probas, y_true, metadata, p=0.8, algo="lr"):
    # 1. Test accuracy.
    print("Computing accuracy...")
    y_pred = np.argmax(probas, axis=1)
    acc = accuracy_score(y_true, y_pred)

    # 2. p-superquantile log loss.
    print("Computing log loss...")
    losses = -np.log(probas)[:, y_true]
    # sq_loss = sq(losses, p)

    # 3. p-superquantile precision across all classes.
    print("Computing precision...")
    prec = precision_score(y_true, y_pred, average=None, zero_division=1)

    # 4. p-superquantile recall across all classes.
    print("Computing recall...")
    rec = recall_score(y_true, y_pred, average=None, zero_division=1)

    # 5. p-superquantile accuracy across all locations
    locations = np.unique(metadata)
    loc_accs = []
    for loc in locations:
        print("Computing accuracy for location %d..." % loc)
        loc_acc = accuracy_score(y_true[metadata==loc], y_pred[metadata==loc])
        loc_accs.append(loc_acc)
    loc_accs = np.array(loc_accs)

    # metrics = {
    #     "accuracy" : acc,
    #     "%0.2f-spq loss" : losses,
    #     "%0.2f-spq precision" : prec,
    #     "%0.2f-spq recall" : rec,
    #     "%0.2f-spq location accuracy" : loc_accs,
    # }

    pickle.dump(acc, open("%s_acc.p" % algo, "wb"))
    pickle.dump(prec, open("%s_prec.p" % algo, "wb"))
    pickle.dump(rec, open("%s_rec.p" % algo, "wb"))
    pickle.dump(losses, open("%s_losses.p" % algo, "wb"))
    pickle.dump(loc_accs, open("%s_loc_accs.p" % algo, "wb"))

    return 0

best_lr_id = pickle.load(open("metrics/best_lr_id.p", "rb"))
coef = pickle.load(open("out/lr_coef_%d.p" % best_lr_id, "rb"))
intercept = pickle.load(open("out/lr_intercept_%d.p" % best_lr_id, "rb"))

X_test = np.load("iwildcam/Z_test.npy")
y_test = np.load("iwildcam/y_test.npy")
metadata = np.load("out/test_meta_0.npy")

np.random.seed(123)
n = len(y_test)
idx = np.random.choice(np.arange(n), 20000, replace=False)

Z = np.dot(X_test[idx], coef.T) + intercept.reshape(1, -1)
lr_probas = softmax(Z)

lr_metrics = compute_metrics(lr_probas, y_test[idx], metadata[idx], p=0.8)



