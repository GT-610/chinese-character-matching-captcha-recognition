import torch
from torch.utils.data import DataLoader
from models.siamese import SiameseNetwork, ContrastiveLoss
from data_process.load_dataset import load_dataset
from torchvision import transforms
from models.siamese import SiameseDataset

def siamese_experiment():
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((45, 45)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # 加载并包装数据集
    base_train = load_dataset(train=True)
    train_dataset = SiameseDataset(base_train, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = SiameseNetwork().to(device)
    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)
    
    # 训练循环
    for epoch in range(20):
        running_loss = 0.0
        for i, (char_imgs, pos_imgs, neg_imgs) in enumerate(train_loader):
            # 转换数据到设备
            char_imgs = char_imgs.to(device)
            pos_imgs = pos_imgs.to(device)
            neg_imgs = neg_imgs.to(device)

            # 正样本对训练
            optimizer.zero_grad()
            output1, output2 = net(char_imgs, pos_imgs)
            loss_pos = criterion(output1, output2, torch.zeros(char_imgs.size(0)).to(device))
            
            # 负样本对训练 
            output1, output2 = net(char_imgs, neg_imgs)
            loss_neg = criterion(output1, output2, torch.ones(char_imgs.size(0)).to(device))
            
            # 合并损失
            total_loss = loss_pos + loss_neg
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
            
        print(f'Epoch {epoch+1} Loss: {running_loss/len(train_loader):.4f}')

    # 在训练循环后添加评估逻辑
    print("\n开始模型评估...")
    base_test = load_dataset(train=False)
    test_dataset = SiameseDataset(base_test, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    evaluate_siamese(net, test_loader, device)  # 传入训练好的模型和测试数据

    return net  # 返回训练好的模型

def evaluate_siamese(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for char_imgs, pos_imgs, _ in test_loader:
            char_imgs = char_imgs.to(device)
            pos_imgs = pos_imgs.to(device)
            
            output1, output2 = model(char_imgs, pos_imgs)
            distances = F.pairwise_distance(output1, output2)
            predictions = (distances < 0.5).long()
            correct += (predictions == 0).sum().item()
            total += predictions.size(0)
    
    accuracy = correct / total
    print(f'验证准确率: {accuracy:.4f}')
    return accuracy
