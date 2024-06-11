import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


def transforms():
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(),
            A.OneOf(
                [
                    A.OpticalDistortion(
                        distort_limit=0.3, border_mode=cv2.BORDER_CONSTANT, value=0
                    ),
                    A.GridDistortion(
                        num_steps=5,
                        distort_limit=0.3,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                    ),
                ],
                p=0.1,
            ),
            A.OneOf([A.Blur(), A.GaussNoise()]),
            A.ColorJitter(),
            A.RandomGamma(),
        ]
    )


def preprocess(config):
    return A.Compose(
        [
            A.LongestMaxSize(config["img_size"]),
            A.Normalize(),
            A.PadIfNeeded(config["img_size"], config["img_size"]),
            ToTensorV2(),
        ]
    )
