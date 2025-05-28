import torch
import torch.nn as nn
from torchvision import models

class ResNetCharClassifier(nn.Module):
    """基于预训练ResNet50的字符分类器（微调版）"""
    def __init__(self, num_classes=9, num_positions=4):
        super().__init__()
        # 加载预训练模型
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # 修改第一层卷积适应单通道输入
        original_conv = self.base_model.conv1
        self.base_model.conv1 = nn.Conv2d(1, original_conv.out_channels, 
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
        # 输入形状: (batch_size, 1, H, W)
        features = self.base_model(x)
        # 调整输出形状为 (batch_size, 4, 9)
        return features.view(-1, self.num_positions, self.num_classes)