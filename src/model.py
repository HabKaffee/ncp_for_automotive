from ncps.torch import LTC
from ncps.wirings import AutoNCP
from src.encoder import Encoder, EncoderResnet

import os

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils as utils

import torchvision as tv
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from sklearn.model_selection import train_test_split
from PIL import Image

from math import ceil

# class CustomDataset(Dataset):
#     def __init__(self, 
#                  annotations_file='out/Town01_opt/data.txt',
#                  img_dir='out/Town01_opt/',
#                  transform=None,
#                  target_transform=None,
#                  sequence_length=10):
#         self.img_dir = img_dir
#         self.image_and_steer = pd.read_csv(annotations_file, sep=":", names=['Image', 'Steer_angle'])
#         assert not self.image_and_steer['Steer_angle'].isna().any(), "Found NaNs in steer angles!"
#         self.image_and_steer = self.image_and_steer.groupby(by=['Image']).Steer_angle.mean().reset_index()
#         self.image_and_steer['Steer_angle'] = pd.to_numeric(self.image_and_steer['Steer_angle'], errors='coerce')
#         self.image_and_steer.dropna(inplace=True)
#         self.image_names = self.image_and_steer['Image'].tolist()
#         self.steer_angles = self.image_and_steer['Steer_angle'].tolist()
#         self.sequence_length = sequence_length
#         self.normalizer = tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

#     def __len__(self):
#         return len(self.image_names) - self.sequence_length + 1
    
#     def __getitem__(self, idx):
#         images = []
#         for i in range(self.sequence_length):
#             img_name = self.image_names[idx + i]
#             image_path = f'{self.img_dir}/{img_name}.png'.replace(" ", "")
#             if not os.path.exists(image_path):
#                 raise FileNotFoundError(f"Missing: {image_path}")
#             image = tv.transforms.functional.pil_to_tensor(Image.open(image_path).convert('RGB'))
#             image = self.normalizer(image / 255)
            
#             # image = torchvision.io.read_image(image_path)[0:3].float() / 255.0
#             # image = torch.clamp(image, 0.0, 1.0)
#             assert not torch.isnan(image).any(), f"NaN in image: {image_path}"
#             images.append(image)

#         images = torch.stack(images, dim=0)
#         steer_angle = self.steer_angles[idx + self.sequence_length - 1]
#         if isinstance(steer_angle, str) or np.isnan(steer_angle):
#             raise ValueError(f"Invalid steer angle at idx {idx}: {steer_angle}")
#         return images, steer_angle

#     def train_test_split(self, test_size=0.2, random_state=42):
#             indices = list(range(len(self.image_names) - self.sequence_length + 1))
#             self.train_indices, self.test_indices = train_test_split(indices,
#                                                         test_size=test_size,
#                                                         random_state=random_state)

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
            # print(df['Steer_angle'].describe())
            image_names = df['Image'].tolist()
            steer_angles = df['Steer_angle'].tolist()
            img_dir = dataset['img_dir']

            group = {
                'image_names': image_names,
                'steer_angles': steer_angles,
                'img_dir': img_dir
            }

            self.data_groups.append(group)

            num_sequences = len(image_names) - sequence_length + 1
            for local_idx in range(num_sequences):
                self.samples.append((group_id, local_idx))

        self.train_indices = []
        self.test_indices = []

    def train_test_split(self, test_size=0.2, random_state=42):
        """
        Generate sequences within each dataset with at most 50% overlap.
        Ensure no shared sequences between train and test.
        """
        self.train_indices = []
        self.test_indices = []

        stride = ceil(self.sequence_length)
        # stride = 1

        for group_id, group in enumerate(self.data_groups):
            image_names = group['image_names']
            max_start = len(image_names) - self.sequence_length + 1
            valid_local_indices = list(range(0, max_start, stride))

            global_indices = []
            for local_idx in valid_local_indices:
                global_idx = self.samples.index((group_id, local_idx)) if (group_id, local_idx) in self.samples else None
                if global_idx is not None:
                    global_indices.append(global_idx)

            train_ids, test_ids = train_test_split(global_indices, test_size=test_size, random_state=random_state)
            self.train_indices.extend(train_ids)
            self.test_indices.extend(test_ids)

    def assign_train_val_by_dataset(self, val_dataset_index):
        """
        Assign one entire dataset for validation, others for training.
        val_dataset_index: index of the dataset in `datasets` list used for validation.
        """
        self.test_indices = [i for i, (gid, _) in enumerate(self.samples) if gid == val_dataset_index]
        self.train_indices = [i for i, (gid, _) in enumerate(self.samples) if gid != val_dataset_index]

        # self.train_indices = []
        # self.test_indices = []
        # for global_idx, (group_id, _) in enumerate(self.samples):
        #     if group_id == val_dataset_index:
        #         self.test_indices.append(global_idx)
        #     else:
        #         self.train_indices.append(global_idx)



    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        group_id, local_idx = self.samples[idx]
        group = self.data_groups[group_id]

        images = []
        for i in range(self.sequence_length):
            img_name = group['image_names'][local_idx + i]
            image_path = os.path.join(group['img_dir'], f"{img_name}.png").replace(" ", "")
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Missing: {image_path}")
            image = tv.transforms.functional.pil_to_tensor(Image.open(image_path).convert('RGB'))
            image = self.normalizer(image / 255.0)
            assert not torch.isnan(image).any(), f"NaN in image: {image_path}"
            images.append(image)

        images = torch.stack(images, dim=0)
        steer_angle = group['steer_angles'][local_idx + self.sequence_length - 1]
        if isinstance(steer_angle, str) or np.isnan(steer_angle):
            raise ValueError(f"Invalid steer angle at idx {idx}: {steer_angle}")
        return images, steer_angle


class Model(nn.Module):
    def __init__(self, output_size, units) -> None:
        super().__init__()
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

        predicted_angle, hx = self.rnn(features, hx, timespans)
        return predicted_angle, hx


class DrivingDataModule(pl.LightningDataModule):
    def __init__(self, dataset: CustomDataset, batch_size: int = 16):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size

    def setup(self, stage=None):
        # Assumes dataset.train_test_split() was called before instantiating the DataModule
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
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer,
        #     mode='min',
        #     factor=0.5,
        #     patience=3,
        #     min_lr=1e-6
        # )
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
    

# class Model(nn.Module):
#     def __init__(self, output_size, units) -> None:
#         super().__init__()
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         #encoder
#         self.encoder = EncoderResnet()

#         self.input_size = self.encoder.output_size
        
#         self.output_size = output_size
#         self.units = units
#         self.rnn = LTC(self.input_size, AutoNCP(self.units, self.output_size), batch_first=True)
#         self.rnn.hx = None

#         self.encoder.to(self.device)
#         self.rnn.to(self.device)

#     def forward(self, input, hx=None, timespans=None, train=False):
#         if not isinstance(input, torch.Tensor):
#             _input = torch.stack(list(input), dim=0)
#             _input = _input.permute(1, 0, 2, 3, 4)
#             print(_input.shape)
#         else:
#             _input = input
#         batch_size, seq_len, c, h, w = _input.shape
#         frames = _input.view(batch_size * seq_len, c, h, w)
#         features = self.encoder(frames)
#         features = features.view(batch_size, seq_len, -1)

#         if self.rnn.hx is not None:
#             self.rnn.hx = self.rnn.hx.detach()
        
#         if train:
#             predicted_angle, self.rnn.hx = self.rnn(features, self.rnn.hx, timespans)
#         else:
#             with torch.no_grad():
#                 predicted_angle, _ = self.rnn(features, self.rnn.hx, timespans)

#         del _input, frames, features
#         return predicted_angle, self.rnn.hx
    
#     def save_model(self, path):
#         torch.save(self.state_dict(), path)
#         torch.save(self.rnn.hx, f'hidden_{path}')
    
#     def load_model(self, state_dict_path):
#         self.load_state_dict(torch.load(state_dict_path))
#         self.rnn.hx = torch.load(f'hidden_{state_dict_path}')

#     def train(self):
#         self.encoder.set_trainable(True)
#         self.rnn.train()

#     def eval(self):
#         self.encoder.set_trainable(False)
#         self.rnn.eval()


# class Trainer:
#     def __init__(self, model, 
#                  loss_func, 
#                  optimizer, 
#                  annotations_file='out/Town01_opt/data.txt', 
#                  img_dir='out/Town01_opt',
#                  test_size=0.2,
#                  random_state=42,
#                  stb_weights=[1, 0, 0],
#                  sequence_length=10,
#                  batch_size=16):
#         self.model = model
#         self.loss_func = loss_func
#         self.optimizer = optimizer
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         self.test_size = test_size
#         self.random_state = random_state
#         self.stb_weights = stb_weights
#         self.sequence_length=sequence_length
#         self.batch_size = batch_size
#         dataset = CustomDataset(annotations_file=annotations_file,
#                                 img_dir=img_dir,
#                                 sequence_length=self.sequence_length)
#         dataset.train_test_split(test_size=self.test_size, random_state=self.random_state)
#         self.train_dataset = dataset
#         self.test_dataset = dataset
#         self.training_start_iso = datetime.now().isoformat()
        
#     def train_one_epoch(self, epoch, logger=None):
#         running_loss = 0.0
#         last_loss = 0.0

#         train_dl = utils.data.DataLoader(self.train_dataset,
#                                          batch_size=self.batch_size,
#                                          sampler=torch.utils.data.SubsetRandomSampler(self.train_dataset.train_indices),
#                                          drop_last=True)
#         idx = 0
#         for data, true_angle in tqdm(train_dl, desc='Train'):
#             true_angle_dev = torch.zeros((true_angle.shape[0], 4), device=self.device)
#             true_angle_dev[:, 0] = true_angle

#             data = data.to(self.device, dtype=torch.float32, non_blocking=True)
#             true_angle_dev = true_angle_dev.to(self.device, dtype=torch.float32, non_blocking=True)
            
#             self.optimizer.zero_grad()
#             prediction, self.model.rnn.hx = self.model(data, train=True)
#             pred_steer = prediction[:, -1, 0]
            
#             steer_loss = self.loss_func(pred_steer, true_angle_dev[:, 0])
#             throttle_loss = self.loss_func(prediction[:, -1, 2], true_angle_dev[:, 2])
#             brake_loss = self.loss_func(prediction[:, -1, 3], true_angle_dev[:, 3])
            
#             loss = steer_loss * self.stb_weights[0] + \
#                     throttle_loss * self.stb_weights[1] + \
#                     brake_loss * self.stb_weights[2]# + \
#                     # reg_magnitude

#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
#             self.optimizer.step()
#             running_loss += loss.item()
#             last_loss = running_loss # loss per batch
#             with open(f'logs/training_{self.training_start_iso}.log', 'a+') as f:
#                 f.write(f'\tItem {idx} of {len(train_dl)} loss: {last_loss}\n')
#             tb_x = epoch * len(self.train_dataset.train_indices) + idx + 1
#             logger.add_scalar('Loss/train', last_loss, tb_x)
#             running_loss = 0.0
#             idx += 1
#         return last_loss

#     def train(self, epochs=10):
        
#         timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#         writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
#         best_val_loss = float('inf')
#         for epoch in range(epochs):
#             print(f'Epoch {epoch}')
#             with open(f'logs/training_{self.training_start_iso}.log', 'a+') as f:
#                 f.write(f'Epoch {epoch}/{epochs} started\n')

#             self.model.train()
#             train_loss = self.train_one_epoch(epoch, writer)
            
#             running_vlos = 0.0
#             self.model.eval()
#             test_dl = utils.data.DataLoader(self.test_dataset,
#                                          batch_size=self.batch_size,
#                                          sampler=torch.utils.data.SubsetRandomSampler(self.test_dataset.test_indices),
#                                          drop_last=True)
#             with torch.no_grad():
#                 for vinputs, vlabels in tqdm(test_dl, desc='Test'):
#                     vinputs = vinputs.to(self.device, dtype=torch.float32)
#                     vlabels = vlabels.to(self.device, dtype=torch.float32)
#                     # vlabels_padded = torch.zeros((vlabels.shape[0], 4), device=self.device)
#                     # vlabels_padded[:, 0] = vlabels
#                     # vlabels_padded[:, 0] = torch.clamp(-vlabels[:, 0], min=0)  # left from steer
#                     # vlabels_padded[:, 1] = torch.clamp(vlabels[:, 0], min=0)   # right from steer
#                     # vlabels_padded[:, 2] = vlabels[:, 1] # throttle
#                     # vlabels_padded[:, 3] = vlabels[:, 2] # brake
#                     voutputs, _ = self.model(vinputs)
#                     # pred_steer = voutputs[:, 1] - voutputs[:, 0]
#                     pred_steer = voutputs[:, -1, 0]
#                     steer_loss = self.loss_func(pred_steer, vlabels)
#                     # throttle_loss = self.loss_func(voutputs[:, -1, 2], vlabels[...])
#                     # brake_loss = self.loss_func(voutputs[:, -1, 3], vlabels[...])
#                     vloss = self.stb_weights[0] * steer_loss # + self.stb_weights[1] * throttle_loss + self.stb_weights[2] * brake_loss
#                     running_vlos += vloss
#             validation_loss = running_vlos / len(self.test_dataset.test_indices)
#             print(f'LOSS train {train_loss} valid {validation_loss}')
#             # if validation_loss < best_val_loss:
#             self.model.save_model(f'model/epoch_{epoch}_{validation_loss:.5f}.pth')
#             # best_val_loss = validation_loss
#             # Log the running loss averaged per batch
#             # for both training and validation
#             writer.add_scalars('Training vs. Validation Loss',
#                     { 'Training' : train_loss, 'Validation' : validation_loss }, 
#                     epoch)
#             writer.flush()
#             with open(f'logs/training_{self.training_start_iso}.log', 'a+') as f:
#                 f.write(f'Epoch {epoch}/{epochs} : LOSS train {train_loss} valid {validation_loss}\n')