import json
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os
import torch
import skimage.io as io
from datasets.transforms import preprocess, transforms
from glob import glob
import pandas as pd
from PIL import Image


class TorchDataset(Dataset):
    def __init__(self, config, mode="train"):
        self.mode = mode
        self.config = config
        self.__init_transforms(config)

        if self.mode == "train":
            self.data = pd.read_csv("data/train.csv")
        else:
            self.data = pd.read_csv("data/test.csv")

    def __init_transforms(self, config):
        self.preprocess = preprocess(config)
        self.transforms = transforms() if self.mode == "train" else None

    def __len__(self):
        return len(self.data)

    def load_sample(self, idx):
        image_name, mask_name = self.data.loc[idx]
        image = io.imread(os.path.join(self.config["data_path"], image_name))
        mask = Image.open(os.path.join(self.config["data_path"], mask_name))
        mask = np.array(mask)
        # print(mask_name, mask.shape)
        return image, mask

    def __getitem__(self, idx):
        image, mask = self.load_sample(idx)
        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        transformed = self.preprocess(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"].to(dtype=torch.long)
        return image, mask


def collate_fn(batch):
    items = list(zip(*batch))
    return [torch.stack(item) for item in items]


def create_loader_dataset(config, mode="train"):

    dataset = TorchDataset(config, mode=mode)
    dataloader = DataLoader(
        dataset,
        shuffle=(mode == "train"),
        collate_fn=collate_fn,
        **config["dataloader"]
    )
    return dataloader, dataset
