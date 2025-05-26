import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import torch.nn.functional as F
import os

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
        img = cv2.imread(os.path.join(sample['captcha_path'], f"{sample['id']}.jpg"))
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
        features_tensor = torch.stack(features)
        
        # 获取候选字符的特征向量
        candidate_images = []
        for i in range(9):  # 修改：加载9个候选字符图像
            candidate_img = cv2.imread(os.path.join(sample['captcha_path'], f"{i}.jpg"), cv2.IMREAD_GRAYSCALE)  # 直接读取为灰度图像
            if candidate_img is None:
                raise FileNotFoundError(f"候选字符图像 {candidate_img_path} 未找到")
            if self.transform:
                candidate_img = self.transform(candidate_img)
            candidate_images.append(candidate_img)
        
        candidate_features_tensor = torch.stack(candidate_images)  # 将候选字符特征转换为张量
        
        # 将标签转换为字符索引数组
        label_str = sample['label']
        label_indices = [int(c) for c in label_str]  # 转换为4位数字索引
        
        return features_tensor, candidate_features_tensor, torch.tensor(label_indices)  # 返回分割特征、候选特征和完整标签序列

class CNNCharClassifier(nn.Module):
    """基于ResNet18的多字符验证码识别模型"""
    def __init__(self, num_classes=9, num_positions=4):
        super().__init__()
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.base_model.fc = nn.Identity()

    def forward(self, x, candidates):
        # 计算候选字符的特征相似度
        batch_size, num_candidates, C, H, W = candidates.shape
        candidates = candidates.view(batch_size * num_candidates, C, H, W)
        candidate_features = self.base_model(candidates)
        candidate_features = candidate_features.view(batch_size, num_candidates, -1)

        # 计算输入图像的特征
        batch_size, num_splits, C, H, W = x.shape  # 使用实际的 C, H, W
        x = x.view(batch_size * num_splits, C, H, W)
        features = self.base_model(x)
        features = features.view(batch_size, num_splits, -1)
        
        # 计算相似度矩阵
        similarity_scores = torch.cosine_similarity(
            features.unsqueeze(2),  # (batch_size, num_splits, 1, feature_dim)
            candidate_features.unsqueeze(1),  # (batch_size, 1, num_candidates, feature_dim)
            dim=-1
        )
        
        return similarity_scores

def train_cnn(model, train_loader, val_loader, epochs=10, device='cuda'):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    
    model.train()
    for epoch in range(epochs):
        for inputs, candidates, labels in train_loader:
            inputs = inputs.to(device)
            candidates = candidates.to(device)
            labels = labels.to(device)  # (batch_size, 4)
            
            optimizer.zero_grad()
            similarity_scores = model(inputs, candidates)  # (batch_size, 4, 9)
            
            # 创建正样本掩码
            batch_size, num_positions, num_classes = similarity_scores.shape
            pos_mask = torch.zeros_like(similarity_scores, dtype=torch.bool)
            pos_mask[torch.arange(batch_size)[:, None], torch.arange(num_positions), labels] = True
            
            # 获取正样本分数（形状保持为 [batch_size, 4]）
            positive_scores = similarity_scores[pos_mask].view(batch_size, num_positions)
            
            # 获取负样本分数（排除正样本后的所有分数）
            negative_scores = similarity_scores[~pos_mask].view(batch_size, num_positions, num_classes-1)
            
            # 计算损失：最大化正样本相似度 + 最小化负样本相似度
            pos_loss = -torch.mean(positive_scores)  # 最大化正样本相似度等价于最小化其负数
            neg_loss = torch.mean(negative_scores)   # 最小化负样本相似度
            loss = pos_loss + neg_loss
            
            loss.backward()
            optimizer.step()
            
        val_acc = evaluate_cnn(model, val_loader, device, verbose=False)  # 新增verbose参数
        scheduler.step(val_acc)

def evaluate_cnn(model, loader, device='cuda', verbose=True):
    model.eval()
    correct = 0
    total = 0
    position_correct = [0]*4
    
    with torch.no_grad():
        for inputs, candidates, labels in loader:
            inputs = inputs.to(device)
            candidates = candidates.to(device)
            labels = labels.to(device)  # (batch_size, 4)
            
            similarity_scores = model(inputs, candidates)  # (batch_size, 4, 9)
            preds = torch.argmax(similarity_scores, dim=2)  # (batch_size, 4)
            
            # 计算整体正确率
            correct += (preds == labels).all(dim=1).sum().item()
            total += labels.size(0)
            
            # 计算各位置正确率
            for i in range(4):
                position_correct[i] += (preds[:, i] == labels[:, i]).sum().item()
    
    if verbose:
        print(f"各位置准确率: {[f'{acc/total:.2%}' for acc in position_correct]}")
    return correct / total

