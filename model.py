import torch
import torch.nn as nn
import torchvision.models as models
from efficientnet_pytorch import EfficientNet


# Define network
#class Model(nn.Module):
#    def __init__(self):
#        super(Model, self).__init__()
#        
#        # Load pre-trained VGG-16 model
#        self.vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
#        
#        # Freeze conv layer parameters
#        for param in self.vgg16.features.parameters():
#            param.requires_grad = False
#        
#        # Modify the classifier for regression
#        self.vgg16.classifier = nn.Sequential(
#            nn.Linear(25088, 4096),
#            nn.ReLU(inplace=True),
#            nn.Dropout(0.5),
#            nn.Linear(4096, 1) 
#        )
#        for layer in self.vgg16.classifier:
#            if isinstance(layer, nn.Linear):
#                nn.init.xavier_uniform_(layer.weight)  # Xavier initialization for linear layers
#                nn.init.constant_(layer.bias, 37.5)  
#        
#    def forward(self, x):
#        out = self.vgg16(x)
#        return out

# Define network
#class Model(nn.Module):
#    def __init__(self):
#        super(Model, self).__init__()
#        
#        # Load pre-trained VGG-16 model
#        self.efficient_net = EfficientNet.from_pretrained('efficientnet-b0')
#        
#        self.efficient_net._fc = nn.Linear(self.efficient_net._fc.in_features, 1) 
#        
#    def forward(self, x):
#        out = self.efficient_net(x)
#        
#        return out

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(57600, 3000),
            nn.ReLU(),
            nn.Linear(3000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, 1) 
        )
        
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight) 
                nn.init.constant_(layer.bias, 37.5)  

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
         
