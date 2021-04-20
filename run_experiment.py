import sys
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from experiment_setup import resnet18, C_hyperparameter

# Get Slurm array job ID.
job_id = sys.argv[1]
C = C_hyperparameter()[job_id]

model = LogisticRegression(C=C)

# Train.
X_train, y_train = resnet18(split="train")
model.fit(X_train, y_train)

# Evaluate.
X_val, y_val = resnet18(split="val")
acc = accuracy_score(y_val, model.predict(X_val))

print("%f %f" % (C, acc))