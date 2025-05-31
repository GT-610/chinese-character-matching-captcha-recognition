import torch
from torch.utils.data import DataLoader
from models.siamese import SiameseNetwork, TripletLoss
from data_process.load_dataset import load_dataset
from torchvision import transforms
from models.siamese import SiameseDataset
import torch.nn.functional as F
import os

def siamese_experiment(force_retrain=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'models/saved/siamese_model.pth'
    
    # 数据预处理（提升到if-else之前）
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((45, 45)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # 检查是否存在预训练模型
    if not force_retrain and os.path.exists(model_path):
        print("加载预训练模型...")
        # 添加安全加载上下文管理器
        with torch.serialization.safe_globals([SiameseNetwork]):
            net = torch.load(
                model_path,
                weights_only=False,  # 启用安全加载模式
                map_location=device
            ).to(device)
    else:
        print("开始训练新模型...")
        # 加载并包装数据集
        base_train = load_dataset(train=True)
        train_dataset = SiameseDataset(base_train, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        # 初始化模型
        net = SiameseNetwork().to(device)
        criterion = TripletLoss(margin=1.0)  # 原为ContrastiveLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)
        
        # 训练循环
        for epoch in range(20):
            running_loss = 0.0
            for i, (char_imgs, pos_imgs, neg_imgs) in enumerate(train_loader):
                # 转换数据到设备
                char_imgs = char_imgs.to(device)
                pos_imgs = pos_imgs.to(device)
                neg_imgs = neg_imgs.to(device)

                # 修改前向传播逻辑
                optimizer.zero_grad()
                
                # 新增三元组损失计算
                anchor = net.forward_once(char_imgs)
                positive = net.forward_once(pos_imgs)
                negative = net.forward_once(neg_imgs)
                total_loss = criterion(anchor, positive, negative)
                
                total_loss.backward()
                optimizer.step()
                
                running_loss += total_loss.item()
                
            print(f'Epoch {epoch+1} Loss: {running_loss/len(train_loader):.4f}')

        # 保存训练好的模型
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(net, model_path)
        print(f"模型已保存至 {model_path}")

    # 评估模型
    print("\n开始模型评估...")
    base_test = load_dataset(train=False)
    test_dataset = SiameseDataset(base_test, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    evaluate_siamese(net, test_loader, device)
    
    return net

def evaluate_siamese(model, test_loader, device):
    model.eval()
    char_correct = 0
    char_total = 0
    captcha_correct = 0
    captcha_total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            # 修改维度重组逻辑：将嵌套维度展开为批量维度
            char_imgs = batch[0].view(-1, 1, 45, 45).to(device)  # [batch*4, 1, 45, 45]
            pos_imgs = batch[1].view(-1, 1, 45, 45).to(device)   # [batch*4, 1, 45, 45]
            
            # 前向传播时保持正确的维度结构
            anchor = model.forward_once(char_imgs)
            positive = model.forward_once(pos_imgs)
            
            # 修改距离计算后的维度恢复逻辑
            distances = F.pairwise_distance(anchor, positive)
            predictions = (distances < 0.5).long()
            
            # 恢复原始验证码结构 [batch_size, 4]
            predictions = predictions.view(-1, 4)  # 将预测结果重组为每个验证码4个字符
            
            # 单字准确率计算（此处逻辑正确）
            char_correct += (predictions == 0).sum().item()
            char_total += predictions.numel()
            
            captcha_pred = (predictions.sum(dim=1) == 0)
            captcha_correct += captcha_pred.sum().item()
            captcha_total += captcha_pred.size(0)

    print(f'单字验证准确率: {char_correct/char_total:.4f}')
    print(f'完整验证码准确率: {captcha_correct/captcha_total:.4f}')
    return (char_correct/char_total, captcha_correct/captcha_total)

