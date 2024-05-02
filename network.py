import torch
import torch.nn as nn
import torchvision.models as models


# Define network
class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()
        # Load pre-trained VGG-16 model
        self.vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        
        # Freeze parameters
        for param in self.vgg16.parameters():
            param.requires_grad = False
        
    def forward(self, x):
        out = self.vgg16(x)
        
        return out
    

class AgePredictor(nn.Module):
    def __init__(self):
        super(AgePredictor, self).__init__()
        
        self.flat = nn.Flatten()
        self.l0 = nn.Linear(1000, 300)
        self.l1 = nn.Linear(300, 300)
        self.l2 = nn.Linear(300, 100)
        self.l3 = nn.Linear(100, 1)
        
        self.body = nn.Sequential(
            self.flat,
            self.l0,
            nn.ReLU(inplace=True),
            self.l1,
            nn.ReLU(inplace=True),
            self.l2,
            nn.ReLU(inplace=True),
            self.l3
        )
    
    def forward(self, x):
        out = self.body(x)
        return out
   
 
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        self.vgg = VGG16FeatureExtractor()
        self.age_predictor = AgePredictor()
        
    def forward(self, x):
        vgg_out = self.vgg(x)
        age_out = self.age_predictor(vgg_out)
        
        return age_out


