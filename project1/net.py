from collections.abc import Callable

import torch
from pytorch_lightning import LightningModule
from torch import nn

torch.set_float32_matmul_precision("medium")


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.net(x) + self.shortcut(x)


def resnet_block(in_channels, out_channels) -> nn.Module:
    return ResNetBlock(in_channels, out_channels)


class ResNet(LightningModule):
    def __init__(
        self,
        num_classes=10,
        input_channels=3,
        block_builder: Callable[[int, int], nn.Module] = resnet_block,
        n_blocks=3,
        block_channels=64,
        hidden_dim=128,
        lr=1e-3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.block_builder = block_builder

        self.lr = lr
        self.n_blocks = n_blocks
        self.block_channels = block_channels

        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            self.block_builder(self.input_channels, self.block_channels),
            nn.ReLU(),
            *[
                nn.Sequential(
                    self.block_builder(self.block_channels, self.block_channels),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                )
                for _ in range(self.n_blocks - 1)
            ],
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.block_channels, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_classes),
        )

        self._num_train_predictions_correct = 0
        self._num_train_predictions_total = 0
        self._num_val_predictions_correct = 0
        self._num_val_predictions_total = 0
        self._num_test_predictions_correct = 0
        self._num_test_predictions_total = 0

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self._num_train_predictions_correct += (logits.argmax(dim=1) == y).sum().item()
        self._num_train_predictions_total += y.size(0)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self._num_val_predictions_correct += (logits.argmax(dim=1) == y).sum().item()
        self._num_val_predictions_total += y.size(0)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        self._num_test_predictions_correct += (logits.argmax(dim=1) == y).sum().item()
        self._num_test_predictions_total += y.size(0)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_train_epoch_end(self):
        train_accuracy = self._num_train_predictions_correct / self._num_train_predictions_total
        self.log("train_accuracy", train_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self._num_train_predictions_correct = 0
        self._num_train_predictions_total = 0

    def on_validation_epoch_end(self):
        val_accuracy = self._num_val_predictions_correct / self._num_val_predictions_total
        self.log("val_accuracy", val_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self._num_val_predictions_correct = 0
        self._num_val_predictions_total = 0

    def on_test_epoch_end(self):
        test_accuracy = self._num_test_predictions_correct / self._num_test_predictions_total
        self.log("test_accuracy", test_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self._num_test_predictions_correct = 0
        self._num_test_predictions_total = 0
