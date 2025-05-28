import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import cv2
from data_process.split_captcha import split_captcha
import os


class SiameseDataset(Dataset):
    """适用于孪生网络的数据集"""
    def __init__(self, base_dataset, transform=None):
        self.base_dataset = base_dataset
        self.transform = transform
        
    def __len__(self):
        return len(self.base_dataset) * 4

    def __getitem__(self, idx):
        sample_idx = idx // 4
        char_pos = idx % 4
        
        sample = self.base_dataset[sample_idx]
        
        img = cv2.imread(os.path.join(sample['captcha_path'], f"{sample['id']}.jpg"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        split_images = split_captcha(img, num_splits=4)
        
        char_img = split_images[char_pos]
        
        # 候选字符加载
        candidate_imgs = []
        for i in range(9):
            candidate_path = os.path.join(sample['captcha_path'], f"{i}.jpg")
            candidate_img = cv2.imread(candidate_path, cv2.IMREAD_GRAYSCALE)
            if candidate_img is None:
                raise FileNotFoundError(f"候选字符图像 {candidate_path} 未找到")
            if self.transform:
                candidate_img = self.transform(candidate_img)
            candidate_imgs.append(candidate_img)
        
        true_label = int(sample['label'][char_pos])
        positive_img = candidate_imgs[true_label]
        negative_img = self._get_negative_sample(candidate_imgs, true_label)
        
        if self.transform:
            char_img = self.transform(char_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)
            
        return (char_img, positive_img, negative_img)

    def _get_negative_sample(self, candidates, true_idx):
        while True:
            idx = torch.randint(0, 9, (1,)).item()
            if idx != true_idx:
                return candidates[idx]

class SiameseNetwork(nn.Module):
    """孪生网络主干结构"""
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        
        # 共享权重的CNN部分
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            
            # 修改第三层卷积核尺寸为4x4，使最终特征图尺寸为5x5
            nn.Conv2d(128, 256, kernel_size=4),  # 原为kernel_size=3
            nn.ReLU(inplace=True)
        )
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(256*5*5, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256)
        )

    def forward_once(self, x):
        x = self.cnn(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

class ContrastiveLoss(nn.Module):
    """对比损失函数"""
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


