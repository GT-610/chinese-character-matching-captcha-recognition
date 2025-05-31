import torch
import torch.nn as nn
from torchvision import models

class ResNetCharClassifier(nn.Module):
    def __init__(self, num_classes=9, num_positions=4):
        super().__init__()
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # 修改输入通道数为4（原为1）
        original_conv = self.base_model.conv1
        self.base_model.conv1 = nn.Conv2d(4, original_conv.out_channels,  # 修改输入通道数
                                        kernel_size=original_conv.kernel_size,
                                        stride=original_conv.stride,
                                        padding=original_conv.padding,
                                        bias=original_conv.bias)
        
        # 替换最后的全连接层
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features, num_classes * num_positions)
        
        # 辅助参数
        self.num_positions = num_positions
        self.num_classes = num_classes

    def forward(self, x):
        # 输入形状修正为: (batch_size, 4, 224, 224)
        # 添加通道维度（如果输入是单通道四位置）
        x = x.view(-1, 4, 224, 224)  # 确保输入维度正确
        features = self.base_model(x)
        return features.view(-1, self.num_positions, self.num_classes)