import torch
from torch.utils.data import DataLoader
from models.siamese import SiameseNetwork, ContrastiveLoss
from data_process.load_dataset import load_dataset
from torchvision import transforms
from models.siamese import SiameseDataset

def train_siamese():
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

def evaluate_siamese(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for char_imgs, pos_imgs, _ in test_loader:  # 负样本在评估时不需要
            char_imgs = char_imgs.to(device)
            pos_imgs = pos_imgs.to(device)
            
            # 计算特征向量
            output1, output2 = model(char_imgs, pos_imgs)
            
            # 计算相似度（使用欧氏距离）
            distances = F.pairwise_distance(output1, output2)
            
            # 判断是否匹配（距离小于阈值设为0.5）
            predictions = (distances < 0.5).long()
            correct += (predictions == 0).sum().item()  # 正样本标签为0
            total += predictions.size(0)
    
    accuracy = correct / total
    print(f'Validation Accuracy: {accuracy:.4f}')
    return accuracy

if __name__ == "__main__":
    train_siamese()