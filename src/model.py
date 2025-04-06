from ncps.torch import LTC
from ncps.wirings import AutoNCP
from src.encoder import Encoder, EncoderResnet

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils as utils

import torchvision
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import train_test_split
from PIL import Image

from datetime import datetime

class TrainingDataset(Dataset):
    def __init__(self, annotations_file='out/Town01_opt/data.txt', img_dir='out/Town01_opt/', transform=None, target_transform=None):
        self.img_dir = img_dir
        self.image_and_steer = pd.read_csv(annotations_file, sep=":", names=['Image', 'Steer_angle'])
        self.image_and_steer = self.image_and_steer.groupby(by=['Image']).Steer_angle.mean().reset_index()
        self.image_names = self.image_and_steer['Image'].tolist()
        self.steer_angles = self.image_and_steer['Steer_angle'].tolist()

    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        image_path = f'{self.img_dir}/{img_name}.png'.replace(" ", "")
        image = torchvision.io.read_image(image_path)[0:3].float() / 255.0

        steer_angle = self.steer_angles[idx]

        return image, steer_angle

    def train_test_split(self, test_size=0.2, random_state=42):
            indices = list(range(len(self.image_names)))
            self.train_indices, self.test_indices = train_test_split(indices,
                                                        test_size=test_size,
                                                        random_state=random_state)
        
    def get_data_by_indices(self, indices):
        images = []
        steer_angles = []
        for idx in indices:
            image, steer_angle = self.__getitem__(idx)
            images.append(image)
            steer_angles.append(steer_angle)
        return images, steer_angles

    def get_train_data(self):
        return self.get_data_by_indices(self.train['indices'])

    def get_test_data(self):
        return self.get_data_by_indices(self.test['indices'])


class Model(nn.Module):
    def __init__(self, output_size, units) -> None:
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #encoder
        self.encoder = EncoderResnet()

        self.input_size = self.encoder.output_size
        
        self.output_size = output_size
        self.units = units
        self.rnn = LTC(self.input_size, AutoNCP(self.units, self.output_size), batch_first=True)
        self.rnn.hx = None

        self.encoder.to(self.device)
        self.rnn.to(self.device)

    def extract_features(self, image : torch.Tensor):
        return self.encoder(image.to(device=self.device, dtype=torch.float32))

    def forward(self, input, hx=None, timespans=None, train=False):
        features = self.encoder(input)
        predicted_angle, self.rnn.hx = self.rnn(features, self.rnn.hx, timespans)
        return predicted_angle, self.rnn.hx
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
        torch.save(self.rnn.hx, f'hidden_{path}')
    
    def load_model(self, state_dict_path):
        self.load_state_dict(torch.load(state_dict_path))
        self.rnn.hx = torch.load(f'hidden_{state_dict_path}')

    def train(self):
        self.encoder.set_trainable(True)
        self.rnn.train()

    def eval(self):
        self.encoder.set_trainable(False)
        self.rnn.eval()


class Trainer:
    def __init__(self, model, 
                 loss_func, 
                 optimizer, 
                 annotations_file='out/Town01_opt/data.txt', 
                 img_dir='out/Town01_opt',
                 test_size=0.2,
                 random_state=42,
                 stb_weights = [1, 0, 0]):
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.Dataset = TrainingDataset(annotations_file=annotations_file,
                                       img_dir=img_dir)
        self.test_size = test_size
        self.random_state = random_state
        self.stb_weights = stb_weights
        
    def train_one_epoch(self, epoch, logger=None, batch_size=1):
        running_loss = 0.0
        last_loss = 0.0

        self.Dataset.train_test_split(test_size=self.test_size, random_state=self.random_state)

        train_dl = utils.data.DataLoader(self.Dataset,
                                         batch_size=batch_size,
                                         sampler=torch.utils.data.SubsetRandomSampler(self.Dataset.train_indices))
        idx = 0
        for image, true_angle in tqdm(train_dl, desc='Train'):
            true_angle_dev = torch.zeros((true_angle.shape[0], 4), device=self.device)
            true_angle_dev[:, 0] = true_angle


            image = image.to(self.device, dtype=torch.float32, non_blocking=True)
            true_angle_dev = true_angle_dev.to(self.device, dtype=torch.float32, non_blocking=True)
            
            self.optimizer.zero_grad()
            prediction, self.model.rnn.hx = self.model(image, train=True)
            self.model.rnn.hx = self.model.rnn.hx.detach()
            pred_steer = prediction[:, 1] - prediction[:, 0]
            
            reg_magnitude = 0.001 * torch.mean((prediction[:, 0]**2 + prediction[:, 1]**2))

            steer_loss = self.loss_func(pred_steer, true_angle_dev[:, 0])
            throttle_loss = self.loss_func(prediction[:, 2], true_angle_dev[:, 2])
            brake_loss = self.loss_func(prediction[:, 3], true_angle_dev[:, 3])
            
            loss = steer_loss * self.stb_weights[0] + \
                    throttle_loss * self.stb_weights[1] + \
                    brake_loss * self.stb_weights[2] + \
                    reg_magnitude

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()
            running_loss += loss.item()
            last_loss = running_loss # loss per batch
            with open('training.log', 'a+') as f:
                f.write(f'\tbatch {idx/1000} loss: {last_loss}\n')
            tb_x = epoch * len(self.Dataset.train_indices) + idx + 1
            logger.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.0
            idx += 1
        return last_loss

    def train(self, epochs=10, batch_size=1):
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
        best_val_loss = float('inf')
        for epoch in range(epochs):
            print(f'Epoch {epoch}')
            with open('training.log', 'a+') as f:
                f.write(f'Epoch {epoch}/{epochs} started\n')

            self.model.train()
            train_loss = self.train_one_epoch(epoch, writer, batch_size=batch_size)
            
            running_vlos = 0.0
            self.model.eval()
            test_dl = utils.data.DataLoader(self.Dataset,
                                         batch_size=batch_size,
                                         sampler=torch.utils.data.SubsetRandomSampler(self.Dataset.test_indices))
            with torch.no_grad():
                for vinputs, vlabels in tqdm(test_dl, desc='Test'):
                    vinputs = vinputs.to(self.device, dtype=torch.float32)
                    vlabels = vlabels.to(self.device, dtype=torch.float32)
                    # vlabels_padded = torch.zeros((vlabels.shape[0], 4), device=self.device)
                    # vlabels_padded[:, 0] = vlabels
                    # vlabels_padded[:, 0] = torch.clamp(-vlabels[:, 0], min=0)  # left from steer
                    # vlabels_padded[:, 1] = torch.clamp(vlabels[:, 0], min=0)   # right from steer
                    # vlabels_padded[:, 2] = vlabels[:, 1] # throttle
                    # vlabels_padded[:, 3] = vlabels[:, 2] # brake
                    voutputs, _ = self.model(vinputs)
                    pred_steer = voutputs[:, 1] - voutputs[:, 0]
                    steer_loss = self.loss_func(pred_steer, vlabels)
                    # throttle_loss = self.loss_func(voutputs[:, 2], vlabels[...])
                    # brake_loss = self.loss_func(voutputs[:, 3], vlabels[...])
                    vloss = self.stb_weights[0] * steer_loss # + self.stb_weights[1] * throttle_loss + self.stb_weights[2] * brake_loss
                    running_vlos += vloss
            validation_loss = running_vlos / len(self.Dataset.test_indices)
            print(f'LOSS train {train_loss} valid {validation_loss}')
            # if validation_loss < best_val_loss:
            self.model.save_model(f'model/epoch_{epoch}_{validation_loss:.5f}.pth')
            # best_val_loss = validation_loss
            # Log the running loss averaged per batch
            # for both training and validation
            writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : train_loss, 'Validation' : validation_loss }, 
                    epoch)
            writer.flush()
            with open('training.log', 'a+') as f:
                f.write(f'Epoch {epoch}/{epochs} : LOSS train {train_loss} valid {validation_loss}\n')
