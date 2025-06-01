from ncps.torch import LTC
from ncps.wirings import AutoNCP
from src.encoder import EncoderResnet, LanePretrainedResNet18Encoder

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import torchvision as tv
from torch.utils.data import Dataset

import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, datasets, sequence_length=10):
        """
        datasets: list of dicts with keys 'annotations_file' and 'img_dir'
        """
        self.sequence_length = sequence_length
        self.normalizer = tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
        self.samples = []  # Each item: (group_id, local_start_idx)
        self.data_groups = []  # Stores each dataset’s data separately

        for group_id, dataset in enumerate(datasets):
            df = pd.read_csv(dataset['annotations_file'], sep=":", names=['Image', 'Steer_angle'])
            df = df.groupby(by=['Image']).Steer_angle.mean().reset_index()
            df['Steer_angle'] = pd.to_numeric(df['Steer_angle'], errors='coerce')
            df.dropna(inplace=True)

            df['Steer_angle'] = ((df['Steer_angle'] + 1) / 2) * 140 - 70
            image_names = df['Image'].tolist()
            steer_angles = df['Steer_angle'].tolist()
            img_dir = dataset['img_dir']

            group = {
                'image_names': image_names,
                'steer_angles': steer_angles,
                'img_dir': img_dir
            }

            self.data_groups.append(group)

            max_start = len(image_names) - sequence_length * (sequence_length - 1)
            for local_idx in range(max_start):
                self.samples.append((group_id, local_idx))

        self.train_indices = []
        self.test_indices = []

    def assign_train_val_by_dataset(self, val_dataset_index):
        self.test_indices = [i for i, (gid, _) in enumerate(self.samples) if gid == val_dataset_index]
        self.train_indices = [i for i, (gid, _) in enumerate(self.samples) if gid != val_dataset_index]



    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        group_id, local_idx = self.samples[idx]
        group = self.data_groups[group_id]

        images = []
        for i in range(self.sequence_length):
            img_index = local_idx + i * self.sequence_length
            img_name = group['image_names'][img_index]
            image_path = os.path.join(group['img_dir'], f"{img_name}.png").replace(" ", "")
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Missing: {image_path}")
            image = tv.transforms.functional.pil_to_tensor(Image.open(image_path).convert('RGB'))
            image = self.normalizer(image / 255.0)
            assert not torch.isnan(image).any(), f"NaN in image: {image_path}"
            images.append(image)

        images = torch.stack(images, dim=0)
        steer_angle = group['steer_angles'][local_idx + self.sequence_length * (self.sequence_length - 1)]
        if isinstance(steer_angle, str) or np.isnan(steer_angle):
            raise ValueError(f"Invalid steer angle at idx {idx}: {steer_angle}")
        return images, steer_angle


class Model(nn.Module):
    def __init__(self, output_size, units) -> None:
        super().__init__()
        # self.encoder = LanePretrainedResNet18Encoder('./tusimple_res18.pth')
        self.encoder = EncoderResnet()
        self.input_size = self.encoder.output_size
        self.output_size = output_size
        self.units = units
        self.rnn = LTC(self.input_size, AutoNCP(self.units, self.output_size), batch_first=True)

    def forward(self, x, hx=None, timespans=None):
        if not isinstance(x, torch.Tensor):
            x = torch.stack(list(x), dim=0).permute(1, 0, 2, 3, 4)

        batch_size, seq_len, c, h, w = x.shape
        frames = torch.reshape(x, (batch_size * seq_len, c, h, w))
        features = self.encoder(frames)
        features = torch.reshape(features, (batch_size, seq_len, *features.shape[1:]))
        if hx is not None:
            hx = hx.detach()

        predicted_angle, _hx = self.rnn(features, hx, timespans)
        return predicted_angle, _hx


class DrivingDataModule(pl.LightningDataModule):
    def __init__(self, dataset: CustomDataset, batch_size: int = 16):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = Subset(self.dataset, self.dataset.train_indices)
        self.val_dataset = Subset(self.dataset, self.dataset.test_indices)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)


class DrivingModelModule(pl.LightningModule):
    def __init__(self, model, loss_func, optimizer_cls, optimizer_kwargs, stb_weights):
        super().__init__()
        self.model = model
        self.loss_func = loss_func
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs
        self.stb_weights = stb_weights

    def on_fit_start(self):
        for param in self.model.encoder.parameters():
            param.requires_grad = True
            # param.requires_grad = False
        for param in self.model.rnn.parameters():
            param.requires_grad = True

    def forward(self, x, hx=None, timespans=None):
        return self.model(x, hx=hx, timespans=timespans)

    def training_step(self, batch, batch_idx):
        self.log("Model_units", self.model.units)
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("lr", current_lr, on_epoch=True, prog_bar=True)
        data, true_angle = batch

        # Target padding
        true_angle_dev = torch.zeros((true_angle.shape[0], 4), device=self.device)
        true_angle_dev[:, 0] = true_angle
        # print(data)
        predictions, _ = self(data, hx=None)

        # Loss computation
        pred_steer = predictions[:, -1, 0]
        steer_loss = self.loss_func(pred_steer, true_angle_dev[:, 0])
        # throttle_loss = self.loss_func(predictions[:, -1, 2], true_angle_dev[:, 2])
        # brake_loss = self.loss_func(predictions[:, -1, 3], true_angle_dev[:, 3])

        loss = (
            steer_loss * self.stb_weights[0] #+
            # throttle_loss * self.stb_weights[1] +
            # brake_loss * self.stb_weights[2]
        )

        if torch.isnan(loss) or torch.isinf(loss):
            self.print("Got NaN loss")
            self.print(f"Prediction: {pred_steer}")
            self.print(f"Target: {true_angle_dev[:, 0]}")
            raise ValueError("NaN loss")


        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, true_angle = batch

        predictions, _ = self(data, hx=None)
        pred_steer = predictions[:, -1, 0]
        steer_loss = self.loss_func(pred_steer, true_angle)

        val_loss = steer_loss * self.stb_weights[0]

        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = self.optimizer_cls(self.parameters(), **self.optimizer_kwargs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=15,
            eta_min=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1
            }
        }
