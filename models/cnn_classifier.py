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
    """Dataset for CAPTCHA images containing split characters and multi-character labels"""
    def __init__(self, dataset, transform=None):
        self.samples = dataset
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # Load CAPTCHA image and convert to grayscale
        img = cv2.imread(os.path.join(sample['captcha_path'], f"{sample['id']}.jpg"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Split image into individual characters
        split_images = split_captcha(img, num_splits=4)
        
        # Process split images
        features = []
        for split_img in split_images:
            if self.transform:
                split_img = self.transform(split_img)
            features.append(split_img)
        
        # Convert features to tensor
        features_tensor = torch.stack(features)
        
        # Process candidate characters
        candidate_images = []
        for i in range(9):  # Load 9 candidate characters
            candidate_img = cv2.imread(os.path.join(sample['captcha_path'], f"{i}.jpg"), cv2.IMREAD_GRAYSCALE)
            if candidate_img is None:
                raise FileNotFoundError(f"Candidate image {os.path.join(sample['captcha_path'], f'{i}.jpg')} not found")
            if self.transform:
                candidate_img = self.transform(candidate_img)
            candidate_images.append(candidate_img)
        
        candidate_features_tensor = torch.stack(candidate_images)  # Convert candidate features to tensor
        
        # Convert labels to indices
        label_str = sample['label']
        label_indices = [int(c) for c in label_str]

        return features_tensor, candidate_features_tensor, torch.tensor(label_indices)

class CNNCharClassifier(nn.Module):
    """ResNet18-based model for multi-character CAPTCHA recognition"""
    def __init__(self, num_classes=9, num_positions=4):
        super().__init__()
        # Modify ResNet18 for grayscale input
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.base_model.fc = nn.Identity()

    def forward(self, x, candidates):
        # Process candidate features
        batch_size, num_candidates, C, H, W = candidates.shape
        candidates = candidates.view(batch_size * num_candidates, C, H, W)
        candidate_features = self.base_model(candidates)
        
        # Process input features
        batch_size, num_splits, C, H, W = x.shape
        x = x.view(batch_size * num_splits, C, H, W)
        features = self.base_model(x)
        
        # Calculate cosine similarity
        similarity_scores = torch.cosine_similarity(
            features.unsqueeze(2),  # Shape: (batch_size, num_splits, 1, feature_dim)
            candidate_features.unsqueeze(1),  # Shape: (batch_size, 1, num_candidates, feature_dim)
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
            
            # Calculate contrastive loss
            batch_size, num_positions, num_classes = similarity_scores.shape
            pos_mask = torch.zeros_like(similarity_scores, dtype=torch.bool)
            pos_mask[torch.arange(batch_size)[:, None], torch.arange(num_positions), labels] = True
            
            # Positive samples loss
            positive_scores = similarity_scores[pos_mask].view(batch_size, num_positions)
            pos_loss = -torch.mean(positive_scores)
            
            # Negative samples loss
            negative_scores = similarity_scores[~pos_mask].view(batch_size, num_positions, num_classes-1)
            neg_loss = torch.mean(negative_scores)
            
            loss = pos_loss + neg_loss
            
            loss.backward()
            optimizer.step()
            
        val_acc = evaluate_cnn(model, val_loader, device, verbose=False)
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
            
            # Calculate overall accuracy
            correct += (preds == labels).all(dim=1).sum().item()
            total += labels.size(0)
            
            # Calculate accuracy for each position
            for i in range(4):
                position_correct[i] += (preds[:, i] == labels[:, i]).sum().item()
    
    if verbose:
        print(f"各位置准确率: {[f'{acc/total:.2%}' for acc in position_correct]}")
    return correct / total

