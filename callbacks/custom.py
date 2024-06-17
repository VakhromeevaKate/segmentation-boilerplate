from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
import torch
import os


class SaveOnnxCallback(Callback):
    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # device = pl_module.config["default_device"],
        test_image = torch.randn(
            1,
            3,
            pl_module.config["img_size"],
            pl_module.config["img_size"],
            requires_grad=False,
        ).to("mps")
        torch.onnx.export(
            pl_module.model,
            test_image,
            os.path.join(
                pl_module.config["exp_path"],
                pl_module.config["project"],
                pl_module.config["exp_name"],
                "onnx_model.onnx",
            ),
            training=torch.onnx.TrainingMode.EVAL,
            export_params=True,
            input_names=["input0"],
            output_names=["output0"],
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
        )
        return super().on_train_end(trainer, pl_module)
