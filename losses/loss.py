import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import segmentation_models_pytorch as smp


class TorchLoss(nn.Module):
    def __init__(self, loss_type, ignore_index):
        super(TorchLoss, self).__init__()
        if loss_type == "Dice":
            self.loss = smp.losses.DiceLoss("multiclass", ignore_index=ignore_index)
        elif loss_type == "Tversky":
            self.loss = smp.losses.TverskyLoss("multiclass", ignore_index=ignore_index)
        elif loss_type == "Focal":
            self.loss = smp.losses.FocalLoss("multiclass", ignore_index=ignore_index)
        elif loss_type == "FocalTversky":
            self.loss = FocalTversky(ignore_index=ignore_index)
        else:
            raise "Unnown loss type"

    def forward(self, preds, target):
        return self.loss(preds, target)


class FocalTversky(nn.Module):
    def __init__(self, ignore_index=-100):
        super().__init__()
        self.loss = smp.losses.TverskyLoss("multiclass", ignore_index=ignore_index)

    def forward(self, preds, target):
        return torch.pow(1 + self.loss(preds, target), 0.75)
