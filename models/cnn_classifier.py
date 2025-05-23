import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import torch.nn.functional as F

from data_process.split_captcha import split_captcha

class CharDataset(Dataset):
    """验证码数据集（分割后的单字图像+多字符标签）"""
    def __init__(self, dataset, transform=None):
        self.samples = dataset  # 直接使用原始数据集
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # 加载验证码图像
        img = cv2.imread(sample['captcha_path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换为灰度图像
        
        # 分割验证码图像为四个部分
        split_images = split_captcha(img, num_splits=4)
        
        # 提取每个部分的特征向量
        features = []
        for split_img in split_images:
            if self.transform:
                split_img = self.transform(split_img)
            features.append(split_img)
        
        # 将 features 转换为张量
        features_tensor = torch.stack(features)  # 将列表转换为张量
        
        # 将标签转换为字符索引数组
        label_str = sample['label']
        label_indices = [int(c) for c in label_str]  # 转换为4位数字索引
        
        return features_tensor, torch.tensor(label_indices)  # 返回张量和完整标签序列

class CNNCharClassifier(nn.Module):
    """基于ResNet18的多字符验证码识别模型"""
    def __init__(self, num_classes=9, num_positions=4):
        super().__init__()
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.base_model.fc = nn.Identity()
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, num_classes)
            ) for _ in range(num_positions)
        ])

    def forward(self, x):
        outputs = []
        batch_size, num_splits, _, _ = x.shape
        for i in range(num_splits):
            feature = self.base_model(x[:, i, ...])
            output = self.classifiers[i](feature)
            outputs.append(output)
        return torch.stack(outputs, dim=1)

def train_cnn(model, train_loader, val_loader, epochs=10, device='cuda'):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)  # (batch_size, 4)
            
            optimizer.zero_grad()
            outputs = model(inputs)  # (batch_size, 4, 9)
            
            loss = 0
            for i in range(4):
                loss += criterion(outputs[:, i, :], labels[:, i])
                
            loss.backward()
            optimizer.step()
            
        val_acc = evaluate_cnn(model, val_loader, device)
        scheduler.step(val_acc)
        print(f'Epoch {epoch+1}/{epochs} | Val Acc: {val_acc:.2%}')

def evaluate_cnn(model, loader, device='cuda'):
    model.eval()
    correct = 0
    total = 0
    position_correct = [0]*4
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)  # (batch_size, 4)
            
            outputs = model(inputs)  # (batch_size, 4, 9)
            preds = torch.argmax(outputs, dim=2)  # (batch_size, 4)
            
            # 计算整体正确率
            correct += (preds == labels).all(dim=1).sum().item()
            total += labels.size(0)
            
            # 计算各位置正确率
            for i in range(4):
                position_correct[i] += (preds[:, i] == labels[:, i]).sum().item()
    
    print(f"各位置准确率: {[f'{acc/total:.2%}' for acc in position_correct]}")
    return correct / total
