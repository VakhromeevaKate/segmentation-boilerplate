import torch
from torch import optim
from transformers import get_cosine_schedule_with_warmup
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from attrdict import AttrDict
import math
from models import TorchModel
from losses.loss import TorchLoss
import os
import pandas as pd
from metrics import class_metrics_multi, class_metrics_binary
from collections import OrderedDict
import numpy as np
import segmentation_models_pytorch.metrics as smpm


class TrainModel(pl.LightningModule):
    def __init__(self, config, train_loader, val_loader):
        super(TrainModel, self).__init__()
        self.config = config
        self.num_classes = self.config["model"]["num_classes"]
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_training_steps = math.ceil(
            len(self.train_loader) / len(config["trainer"]["devices"])
        )
        self.model = TorchModel(config["model"])
        self.criterion = TorchLoss(**config["loss"])
        self.save_hyperparameters(AttrDict(config))

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        self.log(
            "loss/train",
            loss,
            on_step=False,
            on_epoch=True,
            rank_zero_only=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        self.log(
            "loss/val",
            loss,
            on_step=False,
            on_epoch=True,
            rank_zero_only=True,
            sync_dist=True,
        )

        tp, fp, fn, tn = smpm.get_stats(
            torch.argmax(y_hat, dim=1),
            y,
            mode="multiclass",
            num_classes=self.num_classes,
            ignore_index=-1,
        )
        result_metrics = class_metrics_multi(tp, fp, fn, tn)
        for metric, result in result_metrics.items():
            if len(result) > 1:
                for i, res in enumerate(result):
                    self.log(f"{metric}/class_{i}", res, on_step=False, on_epoch=True)
                self.log(f"{metric}/mean", result.mean(), on_step=False, on_epoch=True)
            else:
                self.log(
                    f"{metric}/mean", np.mean(result), on_step=False, on_epoch=True
                )
        # return {
        #     'loss': loss.cpu(),
        #     'tp': tp.cpu(),
        #     'fp': fp.cpu(),
        #     'fn': fn.cpu(),
        #     'tn': tn.cpu()
        # }

    def sync_across_gpus(self, tensors):
        tensors = self.all_gather(tensors)
        return torch.cat([t for t in tensors])

    # def on_validation_epoch_end(self, outputs):
    #     # avg_loss
    #     avg_loss = torch.stack([x["loss"] for x in outputs]).mean()

    #     # metrics
    #     result_metrics = class_metrics_multi(out_val["tp"], out_val["fp"], out_val["fn"], out_val["tn"])
    #     for metric, result in result_metrics.items():
    #         if len(result) > 1:
    #             for i, res in enumerate(result):
    #                 self.log(f"{metric}/class_{i}", res, sync_dist=False)
    #             self.log(f"{metric}/mean", result.mean(), sync_dist=False)
    #         else:
    #             self.log(f"{metric}/mean", np.mean(result), sync_dist=False)

    #     log = {"loss": avg_loss}
    #     self.log("loss/val", avg_loss)

    #     return {"loss": avg_loss, "log": log}

    def configure_optimizers(self):
        if self.hparams.optimizer == "adam":
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                **self.config["optimizer_params"],
            )
        elif self.hparams.optimizer == "sgd":
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                momentum=0.9,
                nesterov=True,
                **self.config["optimizer_params"],
            )
        else:
            raise ValueError(f"Unknown optimizer name: {self.hparams.optimizer}")

        scheduler_params = AttrDict(self.hparams.scheduler_params)
        if self.hparams.scheduler == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                patience=scheduler_params.patience,
                min_lr=scheduler_params.min_lr,
                factor=scheduler_params.factor,
                mode=scheduler_params.mode,
                verbose=scheduler_params.verbose,
            )

            lr_scheduler = {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": scheduler_params.target_metric,
            }
        elif self.hparams.scheduler == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.num_training_steps
                * scheduler_params.warmup_epochs,
                num_training_steps=int(
                    self.num_training_steps * self.config["trainer"]["max_epochs"]
                ),
            )

            lr_scheduler = {"scheduler": scheduler, "interval": "step"}
        else:
            raise ValueError(f"Unknown sheduler name: {self.hparams.sheduler}")

        return [optimizer], [lr_scheduler]
