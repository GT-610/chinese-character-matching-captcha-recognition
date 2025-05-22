import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np

class CharDataset(Dataset):
    """单字字符数据集"""
    def __init__(self, dataset, transform=None):
        self.samples = []
        self.transform = transform
        
        # 展开所有单字样本
        for sample in dataset:
            char_indices = list(map(int, sample['label']))
            for idx in char_indices:
                if idx < len(sample['single_char_paths']):
                    self.samples.append({
                        'path': sample['single_char_paths'][idx],
                        'label': idx
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = cv2.imread(sample['path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            img = self.transform(img)
            
        return img, sample['label']

class CNNCharClassifier(nn.Module):
    """基于预训练ResNet18的字符分类器"""
    def __init__(self, num_classes=9):
        super().__init__()
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)

def train_cnn(model, train_loader, val_loader, epochs=10, device='cuda'):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # 降低学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # 新增学习率调度
    
    best_acc = 0.0
    print("\n=== 训练参数 ===")
    print(f"优化器: {optimizer}")
    print(f"初始学习率: {optimizer.param_groups[0]['lr']}")
    print(f"设备: {device}")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        # 验证
        val_acc = evaluate_cnn(model, val_loader, device)
        train_acc = correct / total
        avg_loss = train_loss / len(train_loader)
        
        # 更新学习率
        scheduler.step(val_acc)
        
        # 打印详细指标
        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2%}')
        print(f'Val Acc: {val_acc:.2%} | LR: {optimizer.param_groups[0]["lr"]:.2e}')
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_cnn.pth')

def evaluate_cnn(model, loader, device='cuda'):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct / total