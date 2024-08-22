import torch
import torchvision.models as models
import torch.nn.functional as F

class EncoderResnet18:
    def __init__(self, 
                 model : models.ResNet = models.resnet18, 
                 weigths : models.Weights = models.ResNet18_Weights.IMAGENET1K_V1):
        self.weights = weigths
        self.model = model()
        self.output_size = 512
        # self.output_size = 1
        self.preprocess = self.weights.transforms()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #remove fc layer
        # self.model.fc = torch.nn.Identity()
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 1)
        # self.model.detection_hdead = torch.nn.Conv2d(self.output_size, 1, kernel_size=1)
        self.model.to(self.device)
        
        self.model.train()
        # self.model.eval()

    def preprocess_image(self, image : torch.Tensor):
        return self.preprocess(image)

    def extract_features(self, image : torch.Tensor):
        transformed_image = self.preprocess_image(image)
        self.output_size = transformed_image.shape[1]
        transformed_image = transformed_image.to(self.device)
        return self.model(transformed_image)

'''
convolution head
layer 1: 24 filters, kernel size 5, strides 2
layer 2: 36 filters, kernel size 5, strides 2
layer 3: 48 filters, kernel size 3, strides 2
layer 4: 64 filters, kernel size 3, strides 1
layer 5: 8 filters, kernel size 3, strides 1
'''

class Encoder(torch.nn.Module):
    def __init__(self, delta1=0.5, delta2=0.5, delta3=0.3):
        super().__init__()
        self.layer_1 = torch.nn.Conv2d(3, 3, kernel_size=5, stride=2)
        self.layer_2 = torch.nn.Conv2d(3, 3, kernel_size=5, stride=2)
        self.layer_3 = torch.nn.Conv2d(3, 3, kernel_size=3, stride=2)
        self.layer_4 = torch.nn.Conv2d(3, 3, kernel_size=3, stride=1)
        self.layer_5 = torch.nn.Conv2d(3, 3, kernel_size=3, stride=1)
        self.flatten = torch.nn.Flatten()
        self.dropout_1 = torch.nn.Dropout(p=delta1)
        self.fc_1 = torch.nn.Linear(in_features=76,out_features=1000) 
        self.dropout_2 = torch.nn.Dropout(p=delta2)
        self.fc_2 = torch.nn.Linear(in_features=1000, out_features=100)
        self.dropout_3 = torch.nn.Dropout(p=delta3)
        self.fc_3 = torch.nn.Linear(in_features=100, out_features=10)
        self.ident_layer = torch.nn.Identity()

    def forward(self, x):
        x = self.layer_1(x)
        print(x.shape)
        x = self.layer_2(x)
        print(x.shape)
        x = self.layer_3(x)
        print(x.shape)
        x = self.layer_4(x)
        print(x.shape)
        x = self.layer_5(x)
        print(x.shape)
        x = self.flatten(x)
        print(x.shape)
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
