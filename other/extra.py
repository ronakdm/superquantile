from sklearn.metrics import precision_score, recall_score

def compute_metrics(y_proba, y_true, metadata, ps):
    
    # 1. Test accuracy.
    y_pred = np.argmax(y_proba, axis=1)
    acc = accuracy_score(y_true, y_pred)

    # 2. Log losses.
    print("Computing log loss...")
    n = len(y_true)
    losses = np.zeros(n)
    for i in range(n):
        losses[i] = -np.log(y_proba)[i, class_to_idx[y_true[i]]]

    # 3. Precision across all classes.
    print("Computing precision...")
    prec = precision_score(y_true, y_pred, average=None, zero_division=0)

    # 4. Recall across all classes.
    print("Computing recall...")
    rec = recall_score(y_true, y_pred, average=None, zero_division=0)

    # 5. Accuracy across all locations
    print("Computing accuracy for each location...")
    locations = np.unique(metadata)
    loc_accs = []
    for loc in locations:
        loc_acc = 1 - accuracy_score(y_true[metadata==loc], y_pred[metadata==loc])
        loc_accs.append(loc_acc)
    loc_accs = np.array(loc_accs)
    print(loc_accs)
    
    sq_loss = []
    sq_prec = []
    sq_rec = []
    sq_loc = []
    for p in ps:
        sq_loss.append(sq(losses, p))
        sq_prec.append(sq(prec, p))
        sq_rec.append(sq(rec, p))
        sq_loc.append(sq(loc_accs, p))
        
    return acc, sq_loss, sq_prec, sq_rec, sq_loc

# fig, axes = plt.subplots(1, 2, figsize=(20, 10))
# axes[0].plot(ps, lr_sq_prec, color="red", linestyle = "--", linewidth=3, label="LR p-SQ Precision")
# axes[0].plot(ps, lr_sq_rec, color="red", linewidth=3, label="LR p-SQ Recall")
# axes[1].plot(ps, lr_sq_loss, color="red", linewidth=3, label="LR p-SQ log-loss")

# axes[0].set_label("p")
# axes[1].set_label("p")

# axes[0].plot(ps, drlr_sq_prec, color="blue", linestyle = "--", linewidth=3, label="DRLR p-SQ Precision")
# axes[0].plot(ps, drlr_sq_rec, color="blue", linewidth=3, label="DRLR p-SQ Recall")
# axes[1].plot(ps, drlr_sq_loss, color="blue", linewidth=3, label="DRLR p-SQ log-loss")

# axes[0].legend()
# axes[1].legend()

# plt.show()

fig, ax = plt.subplots(1, 1, figsize=(10, 10))

ax.plot(ps, lr_sq_loc, color="red", linewidth=3, label="LR p-SQ Location-Based Error SQ")
ax.set_label("p")
ax.plot(ps, drlr_sq_loc, color="blue", linewidth=3, label="DRLR p-SQ Location-Based Error SQ")
ax.legend()

plt.show()