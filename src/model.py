from ncps.torch import LTC
from ncps.wirings import AutoNCP
from src.encoder import Encoder, EncoderResnet18

import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.io import read_image
from torchvision.transforms.functional import pil_to_tensor
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import train_test_split
from PIL import Image

from datetime import datetime

# from src.encoder import EncoderResnet18

class Model(nn.Module):
    def __init__(self, output_size, units) -> None:
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #encoder
        self.encoder = Encoder()
        self.input_size = self.encoder.fc_3.out_features
        # self.encoder = EncoderResnet18().model
        # self.input_size = self.encoder.fc.in_features
        # self.encoder_weights = models.ResNet18_Weights.IMAGENET1K_V1
        # self.encoder.fc = torch.nn.Identity()
        # self.encoder_preprocess = self.encoder_weights.transforms()
        
        self.output_size = output_size
        self.units = units
        self.rnn = LTC(self.input_size, AutoNCP(self.units, self.output_size), batch_first=True)
        
        # self.encoder.to(self.device)
        self.encoder.to(self.device)
        self.rnn.to(self.device)

    def extract_features(self, image : torch.Tensor):
        # transformed_image = self.encoder_preprocess(image)
        # self.input_size = transformed_image.shape[1]
        image = image.to(device=self.device, dtype=torch.float)
        image = image.to(device=self.device, dtype=torch.float)
        # transformed_image = transformed_image.to(self.device)
        # x = self.encoder(transformed_image)
        # x = self.encoder.segmentation_head(x)
        # x = nn.functional.interpolate(x, scale_factor=32, mode='bilinear', align_corners=False)
        return self.encoder(image)
        # return self.encoder(transformed_image)

    def forward(self, input, hx=None, timespans=None):
        features = self.extract_features(input)
        # features.to(self.device)
        return self.rnn(features, hx, timespans)
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
    
    def load_model(self, state_dict_path):
        self.load_state_dict(torch.load(state_dict_path))

    def train(self):
        self.encoder.train(True)
        self.rnn.train(True)

    def eval(self):
        self.encoder.eval()
        self.rnn.eval()

class TrainingDataset(Dataset):
    def __init__(self, annotations_file='out/Town01_opt/data.txt', img_dir='out/Town01_opt/', transform=None, target_transform=None):
        self.image_and_steer = pd.read_csv(annotations_file, sep=":", names=['Image', 'Steer_angle'])
        self.image_and_steer = self.image_and_steer.groupby(by=['Image']).Steer_angle.mean().reset_index()
        self.image_and_steer = self.image_and_steer.to_dict('list')
        image_names = self.image_and_steer['Image']
        for idx, img_name in enumerate(image_names):
            image = Image.open(f'{img_dir}/{img_name}.png')
            self.image_and_steer['Image'][idx] = image

    def train_test_split(self, test_size=0.2, random_state=42):
        to_split = []
        for item in pd.DataFrame.from_records(self.image_and_steer).iterrows():
            to_split.append((item[1]['Image'], item[1]['Steer_angle']))
        print(to_split)
        self.train, self.test = train_test_split(to_split,
                                                 test_size=test_size,
                                                 random_state=random_state)
        return self.train, self.test

class Trainer:
    def __init__(self, model, 
                 loss_func, 
                 optimizer, 
                 annotations_file='out/Town01_opt/data.txt', 
                 img_dir='out/Town01_opt',
                 test_size=0.2,
                 random_state=42):
        self.model = model
        self.model_encoder = self.model.encoder
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch.set_grad_enabled(True)
        self.Dataset = TrainingDataset(annotations_file=annotations_file,
                                       img_dir=img_dir)
        self.test_size = test_size
        self.random_state = random_state
        
    def train_one_epoch(self, epoch, logger=None):
        running_loss = 0
        last_loss = 0
        train, test = self.Dataset.train_test_split(test_size=0.2, random_state=self.random_state)
        for i, data in enumerate(train):
            image, true_angle = data
            image = pil_to_tensor(image)
            # print(f'Img={image}, angl = {true_angle}')
            pred_angle = self.model(image)
            print(pred_angle[0].shape)
            loss = self.loss_func(pred_angle, true_angle)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            if i % 100 == 0:
                last_loss = running_loss / 1000 # loss per batch
                print(f'  batch {i+1} loss: {last_loss}')
                tb_x = epoch * len(train) + i + 1
                logger.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.   
        return last_loss
         

    def train(self, epochs=10):
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))

        for epoch in range(epochs):
            print(f'Epoch {epoch}')
            self.model.train()
            train_loss = self.train_one_epoch(epoch, writer)
            
            running_vlos = 0.0
            self.model.eval()
            with torch.no_grad():
                for i, vdata in enumerate(self.Dataset.test):
                    vinputs, vlabels = vdata
                    vinputs = pil_to_tensor(vinputs)
                    voutputs = self.model(vinputs)
                    vloss = self.loss_func(voutputs, vlabels)
                    running_vloss += vloss
            validation_loss = running_vloss / (i + 1)
            print(f'LOSS train {train_loss} valid {validation_loss}')

            # Log the running loss averaged per batch
            # for both training and validation
            writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : train_loss, 'Validation' : validation_loss }, 
                    epoch)
            writer.flush()
