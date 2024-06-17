#!/usr/bin/env python3
import os
import sys
import argparse
import yaml
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_only
from datasets.train import create_loader_dataset
from pl_models import TrainModel
import shutil
from loguru import logger
from callbacks.custom import SaveOnnxCallback



def load_config(config_path):
    with open(config_path, "r") as input_file:
        config = yaml.safe_load(input_file)

    return config

@rank_zero_only
def check_dir(dirname):
    if not os.path.exists(dirname):
        return

    print(f"Save directory - {dirname} exists")
    print("Ignore: Yes[y], No[n]")
    ans = input().lower()
    if ans == "y":
        shutil.rmtree(dirname)
        return
    raise ValueError("Tried to log experiment into existing directory")


def convert_model_to_onnx(model):
    config = load_config("configs/train.yaml")
    filepath = config["model_exp_path"]
    input_sample = torch.randn((1, 64))
    model.to_onnx(filepath, input_sample, export_params=True)


@logger.catch
def train(args=None):

    config = load_config("configs/train.yaml")
    config["save_path"] = os.path.join(
        config["exp_path"], config["project"], config["exp_name"]
    )

    default_device = config["default_device"]

    if default_device == "cuda":
        torch.cuda.empty_cache()

    torch.set_default_device(default_device)

    check_dir(config["save_path"])
    os.makedirs(config["save_path"], exist_ok=True)

    tensorboard_logger = TensorBoardLogger(config["save_path"], name="metrics")

    train_loader, _ = create_loader_dataset(config)

    val_loader, _ = create_loader_dataset(config, mode="test")

    model = TrainModel(config, train_loader, val_loader)

    checkpoint_callback = ModelCheckpoint(
        dirpath=config["save_path"],
        save_last=True,
        every_n_epochs=1,
        save_top_k=1,
        save_weights_only=True,
        save_on_train_epoch_end=False,
        **config["checkpoint"],
    )

    callbacks = [
        LearningRateMonitor(logging_interval="epoch"),
        checkpoint_callback,
        SaveOnnxCallback(),
    ]

    trainer = Trainer(
        callbacks=callbacks,
        logger=tensorboard_logger,
        log_every_n_steps=len(train_loader),
        **config["trainer"],
    )
    trainer.fit(model)


if __name__ == "__main__":
    train()
