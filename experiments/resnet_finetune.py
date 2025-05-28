import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from models.resnet_finetune import ResNetCharClassifier
from data_process.load_dataset import load_dataset
from models.cnn_classifier import CharDataset  # 复用现有数据集

def resnet_finetune_experiment():
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # 加载数据集
    train_set = CharDataset(load_dataset(train=True), transform=transform)
    test_set = CharDataset(load_dataset(train=False), transform=transform)
    
    # 创建数据加载器
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
    
    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNetCharClassifier(num_classes=10, num_positions=4).to(device)
    
    # 训练配置
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # 训练循环
    for epoch in range(15):
        model.train()
        total_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # 计算每个位置的损失
            loss = 0
            for i in range(4):
                loss += criterion(outputs[:, i, :], labels[:, i])
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # 验证评估
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