import os
import numpy as np
from sklearn.metrics import confusion_matrix

def show_metrics(path = "./output"):
    pids = [540, 544, 552, 567, 584, 596]
    threshold = 80
    res = []
    for pid in pids:
        arr = np.loadtxt(os.path.join(path, f"{pid}.txt"))
        true = arr[:, 0]
        pred = arr[:, 1]
        pred_label = (pred < threshold).astype(int)
        true_label = (true < threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(true_label, pred_label).ravel()
        accuracy = (tn+tp)/(tn+fp+fn+tp)
        sensitivity = tp/(tp+fn)
        precision=tp/(tp+fp)
        f1 = tp/(tp+1/2*(fp+fn))
        specificity = tn/(tn+fp)
        npv = tn/(tn+fn)
        print(pid, accuracy, sensitivity, precision)
        res.append([accuracy, sensitivity, precision, f1, specificity, npv])
    res = np.array(res)
    print(np.mean(res, axis=0))
