import torch
from torchmetrics import PrecisionRecallCurve
from sklearn.metrics import recall_score, precision_score
import segmentation_models_pytorch.metrics as smpm
import numpy as np


def accuracy(y, y_hat):
    y_hat = torch.argmax(y_hat, dim=1)
    acc = (y == y_hat).sum() / y.size(0)
    return acc.item()


def class_metrics_binary(y, y_hat):
    y_hat = torch.argmax(y_hat, dim=1)
    confusion_vector = y_hat / y
    TP = torch.sum(confusion_vector == 1).item()
    FP = torch.sum(confusion_vector == float("inf")).item()
    TN = torch.sum(torch.isnan(confusion_vector)).item()
    FN = torch.sum(confusion_vector == 0).item()

    acc = (TP + TN) / (TP + TN + FP + FN + 0.0001)
    specificity = TN / (TN + FP + 0.0001)
    precision = TP / (TP + FP + 0.0001)
    recall = TP / (TP + FN + 0.0001)
    return acc, specificity, precision, recall


def class_metrics_multi(tp, fp, fn, tn):
    # output = torch.argmax(output, dim=1)
    # tp, fp, fn, tn = smpm.get_stats(output, target, mode='multiclass', num_classes=8)
    result = {
        "iou": smpm.iou_score(tp, fp, fn, tn).mean(0),
        "b-acc": [smpm.balanced_accuracy(tp, fp, fn, tn).mean()],
        "f1": smpm.f1_score(tp, fp, fn, tn).mean(0),
        "pres": smpm.precision(tp, fp, fn, tn).mean(0),
        "rec": smpm.recall(tp, fp, fn, tn).mean(0),
        "specificity": smpm.specificity(tp, fp, fn, tn).mean(0),
        "sensitivity": smpm.sensitivity(tp, fp, fn, tn).mean(0),
        "acc": [smpm.accuracy(tp, fp, fn, tn).mean()],
    }
    return result
