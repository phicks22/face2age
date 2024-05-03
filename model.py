import torch
import torch.nn as nn
import torchvision.models as models


# Define network
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        # Load pre-trained VGG-16 model
        self.vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        
        # Freeze conv layer parameters
        for param in self.vgg16.features.parameters():
            param.requires_grad = False
        
        # Modify the classifier for regression
        self.vgg16.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1) 
        ) 
        
    def forward(self, x):
        out = self.vgg16(x)
        
        return out
    
