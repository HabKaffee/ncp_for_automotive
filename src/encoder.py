import torch
import torchvision.models as models
import torch.nn.functional as F
import numpy as np

class EncoderResnet(torch.nn.Module):
    def __init__(self, 
                 model_fn : models.ResNet = models.resnet18, 
                 weigths : models.Weights = models.ResNet18_Weights.DEFAULT,
                 train_encoder : bool = True):
        super(EncoderResnet, self).__init__()
        self.weights = weigths
        self.model = model_fn(weights=self.weights)
        self.output_size = 512
        self.preprocess = self.weights.transforms()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Remove classification layer
        self.model.fc = torch.nn.Identity()
        self.model.to(self.device)
        self.set_trainable(train_encoder)
        self.freeze_batchnorm()

    def freeze_batchnorm(self):
        for module in self.model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

    def forward(self, data):
        if data.dim() != 4:
            raise ValueError(f"Expected 4D input (B, C, H, W), got {data.shape}")

        return self.model(data)

    def set_trainable(self, trainable: bool):
        """
        Toggle the trainability of the encoder dynamically.
        """
        for param in self.model.parameters():
            param.requires_grad = trainable

'''
convolution head
layer 1: 24 filters, kernel size 5, strides 2
layer 2: 36 filters, kernel size 5, strides 2
layer 3: 48 filters, kernel size 3, strides 2
layer 4: 64 filters, kernel size 3, strides 1
layer 5: 8 filters, kernel size 3, strides 1
'''

class Encoder(torch.nn.Module):
    '''
    Legacy encoder. Do not use!
    '''
    def __init__(self, delta1=0.5, delta2=0.5, delta3=0.3):
        super().__init__()
        self.layer_1 = torch.nn.Conv2d(3, 3, kernel_size=5, stride=2)
        self.layer_2 = torch.nn.Conv2d(3, 3, kernel_size=5, stride=2)
        self.layer_3 = torch.nn.Conv2d(3, 3, kernel_size=3, stride=2)
        self.layer_4 = torch.nn.Conv2d(3, 3, kernel_size=3, stride=1)
        self.layer_5 = torch.nn.Conv2d(3, 3, kernel_size=3, stride=1)
        self.flatten = torch.nn.Flatten()
        self.dropout_1 = torch.nn.Dropout(p=delta1)
        self.fc_1 = torch.nn.Linear(in_features=228,out_features=1000) 
        self.dropout_2 = torch.nn.Dropout(p=delta2)
        self.fc_2 = torch.nn.Linear(in_features=1000, out_features=100)
        self.dropout_3 = torch.nn.Dropout(p=delta3)
        self.fc_3 = torch.nn.Linear(in_features=100, out_features=10)
        self.ident_layer = torch.nn.Identity()

    def forward(self, x):
        # print(x.shape)
        x = self.layer_1(x)
        # print(x.shape)
        x = self.layer_2(x)
        # print(x.shape)
        x = self.layer_3(x)
        # print(x.shape)
        x = self.layer_4(x)
        # print(x.shape)
        x = self.layer_5(x)
        # print(x.shape)
        x = self.flatten(x)
        # print(x.shape)
        x = self.dropout_1(x)
        x = self.fc_1(x)
        x = F.relu(x)
        x = self.dropout_2(x)
        x = self.fc_2(x)
        x = F.relu(x)
        x = self.dropout_3(x)
        x = self.fc_3(x)
        x = self.ident_layer(x)
        return x
