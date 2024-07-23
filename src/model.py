from ncps.torch import LTC
from ncps.wirings import AutoNCP

import torch
import torch.nn as nn
import torchvision.models as models

# from src.encoder import EncoderResnet18

class Model(nn.Module):
    def __init__(self, output_size, units) -> None:
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #encoder
        self.encoder = models.resnet18()
        self.input_size = 512 #self.encoder.fc.in_features
        self.encoder_weights = models.ResNet18_Weights.IMAGENET1K_V1
        self.encoder.fc = torch.nn.Identity()
        self.encoder_preprocess = self.encoder_weights.transforms()
        
        self.output_size = output_size
        self.units = units
        self.rnn = LTC(self.input_size, AutoNCP(self.units, self.output_size), batch_first=True)
        
        self.encoder.to(self.device)
        self.rnn.to(self.device)

    def extract_features(self, image : torch.Tensor):
        transformed_image = self.encoder_preprocess(image)
        self.input_size = transformed_image.shape[1]
        transformed_image = transformed_image.to(self.device)
        # x = self.encoder(transformed_image)
        # x = self.encoder.segmentation_head(x)
        # x = nn.functional.interpolate(x, scale_factor=32, mode='bilinear', align_corners=False)
        return self.encoder(transformed_image)

    def forward(self, input, hx=None, timespans=None):
        features = self.extract_features(input)
        # features.to(self.device)
        return self.rnn(features, hx, timespans)
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
    
    def load_model(self, state_dict_path):
        self.load_state_dict(torch.load(state_dict_path))

    def train(self):
        # self.encoder.train()
        self.encoder.eval()
        self.rnn.train()

    def eval(self):
        self.encoder.eval()
        self.rnn.eval()

class Trainer:
    def __init__(self, model, loss_func, optimizer):
        self.model = model
        self.model_encoder = self.model.encoder
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.model.to(self.device)

    def train(self, data, true_angle):
        self.model.train()
        data = data.to(self.device)
        true_angle = true_angle.to(self.device)
        
        line_extraction = self.model.extract_features(data)

        prediction, _ = self.model.forward(data)
        prediction = prediction[0]#torch.mean(prediction[0])
        # loss = torch.sqrt(self.loss_func(prediction, true_angle))
        loss = self.loss_func(prediction.item(), true_angle)
        
        #back propagation
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        print(f'Current loss (*1000) = {loss*1000:.5f}, predicted = {prediction.item():.7f}, true = {true_angle:.7f}')
