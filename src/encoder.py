import torch
import torchvision.models as models

class EncoderResnet18:
    def __init__(self, 
                 model : models.ResNet = models.resnet18, 
                 weigths : models.Weights = models.ResNet18_Weights.IMAGENET1K_V1):
        self.weights = weigths
        self.model = model()
        self.output_size = 512
        self.preprocess = self.weights.transforms()
        #remove fc layer
        self.model.fc = torch.nn.Identity()
        self.model.eval()

    def preprocess_image(self, image : torch.Tensor):
        return self.preprocess(image)

    def extract_features(self, image : torch.Tensor):
        transformed_image = self.preprocess_image(image)
        self.output_size = transformed_image.shape[1]
        return self.model(transformed_image)

