import torch.nn as nn
import torch
import timm
from copy import deepcopy
from timm import create_model
import torchvision
import segmentation_models_pytorch as smp


class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()

        self.model = smp.create_model(
            config["model_type"],
            encoder_name=config[
                "encoder_name"
            ],  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=config[
                "in_channels"
            ],  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=config[
                "num_classes"
            ],  # model output channels (number of classes in your dataset)
        )

    def forward(self, x):
        return self.model(x)
