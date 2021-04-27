from sklearn.decomposition import PCA
import numpy as np
from experiment_setup import resnet18
import time
import datetime

def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

X_full, y_full = resnet18(split="train", verbose=True)

n = 16000
d_max = X_full.shape[1]

pca = PCA(n_components=d_max)

# Subset.
np.random.seed(123)
idx = np.random.permutation(X_full.shape[0])
X, y = X_full[idx[0:n], :], y_full[idx[0:n]]

# Fit on training data and time.

tic = time.time()
pca.fit(X)
toc = time.time()

print("Fit time: {:}.".format(format_time(toc-tic)))

cumulative = 0
for i, var in enumerate(pca.explained_variance_ratio_):
    cumulative += var
    if cumulative > 0.95:
        print("%0.3f percent of variance explained by first %d dimensions." % (cumulative, i))
        break

# Apply to validation and test data.

tic = time.time()
Z_train = pca.transform(X)
Z_val = pca.transform(resnet18(split="val", verbose=True)[0])
Z_test = pca.transform(resnet18(split="test", verbose=True)[0])
toc = time.time()

print("Transform time: {:}.".format(format_time(toc-tic)))

with open("iwildcam/Z_train.npy", "wb") as f:
    np.save(f, Z_train)
with open("iwildcam/Z_val.npy", "wb") as f:
    np.save(f, Z_val)
with open("iwildcam/Z_test.npy", "wb") as f:
    np.save(f, Z_test)




