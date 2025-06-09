import torch
import torch.nn as nn
from torchvision import models

class ResNetCharClassifier(nn.Module):
    def __init__(self, num_classes=9, num_positions=4):
        super().__init__()
        # Load pre-trained ResNet50 with updated weights
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # Modify first convolution layer to accept 4 channels input
        original_conv = self.base_model.conv1
        self.base_model.conv1 = nn.Conv2d(4, original_conv.out_channels,
                                        kernel_size=original_conv.kernel_size,
                                        stride=original_conv.stride,
                                        padding=original_conv.padding,
                                        bias=original_conv.bias)
        
        # Replace final fully connected layer
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features, num_classes * num_positions)
        
        # Auxiliary parameters for model configuration
        self.num_positions = num_positions
        self.num_classes = num_classes

    def forward(self, x):
        # Reshape input tensor to (batch_size, channels, height, width)
        x = x.view(-1, 4, 224, 224)
        features = self.base_model(x)
        return features.view(-1, self.num_positions, self.num_classes)