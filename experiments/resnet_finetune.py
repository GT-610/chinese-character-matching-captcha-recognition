import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from models.resnet_finetune import ResNetCharClassifier
from data_process.load_dataset import load_dataset
from data_process.split_captcha import split_captcha
from models.cnn_classifier import CharDataset
import torch.nn as nn
import cv2

def resnet_finetune_experiment():
    # Data preprocessing
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Load datasets for training and testing
    train_set = CharDataset(load_dataset(train=True), transform=transform)
    test_set = CharDataset(load_dataset(train=False), transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
    
    # Initialize the model and move it to the appropriate device (GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNetCharClassifier(num_classes=10, num_positions=4).to(device)
    
    # Training configuration
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(15):
        model.train()
        total_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Calculate loss for each position
            loss = 0
            for i in range(4):
                loss += criterion(outputs[:, i, :], labels[:, i])
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Validation evaluation
        val_acc = evaluate_resnet(model, test_loader, device)
        print(f'Epoch {epoch+1}/15 | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2%}')
    
    return model

def evaluate_resnet(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=2)
            correct += (preds == labels).all(dim=1).sum().item()
            total += labels.size(0)
    
    return correct / total

# Dataset class for character images
class CharDataset(Dataset):
    def __init__(self, base_dataset, transform=None):
        self.base_dataset = base_dataset
        self.transform = transform
        
    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        img = cv2.imread(sample['captcha_path'] + f"/{sample['id']}.jpg", 0)
        split_images = split_captcha(img, num_splits=4)
        
        # Process four character images
        char_imgs = []
        labels = []
        for i in range(4):
            char_img = split_images[i]
            if self.transform:
                char_img = self.transform(char_img)
            char_imgs.append(char_img)
            labels.append(int(sample['label'][i]))
            
        # Combine four single-channel images into a four-channel input
        char_imgs = torch.cat(char_imgs, dim=0)  # [4, 224, 224]
        
        # Add channel dimension and transpose dimensions to [C, H, W]
        char_imgs = char_imgs.unsqueeze(1).permute(1, 0, 2, 3)  # [1, 4, 224, 224]
        
        return char_imgs, torch.LongTensor(labels)