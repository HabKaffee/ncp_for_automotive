import torch
import torchvision.models as models

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
        # self.model.detection_head = torch.nn.Conv2d(self.output_size, 1, kernel_size=1)
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

class Encoder(torch.nn):
    def __init__(self):
       self.layer_1 = torch.nn.conv2d() 
