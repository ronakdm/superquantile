import numpy as np
# from utils import resnet18
from sklearn.decomposition import PCA

# X = {}
# y = {}

# for split in ["train", "val", "test"]:

#     print("Loading and subsampling %s data..." % split)

#     idx = np.load("%s_idx.npy" % split)
#     X_, y_ = resnet18(split=split, verbose=True)
#     X[split], y[split] = X_[idx], y_[idx]
#     np.save("data/X_%s_reduced.npy" % split, X[split])
#     np.save("data/y_%s_reduced.npy" % split, y[split])

X_train = np.load("data/X_train_reduced.npy")

pca = PCA()
pca.fit(X_train)

cumulative = 0
for i, var in enumerate(pca.explained_variance_ratio_):
    cumulative += var
    if cumulative > 0.95:
        print("%0.3f percent of variance explained by first %d dimensions." % (cumulative, i))
        n_components = i
        break

pca = PCA(n_components=n_components).fit(X_train)

for split in ["train", "val", "test"]:
    X = np.load("data/X_%s_reduced.npy" % split)
    np.save("data/X_%s_reduced.npy" % split, pca.transform(X))


