import torch
from torch.utils.data import DataLoader
from models.siamese import SiameseNetwork, TripletLoss
from data_process.load_dataset import load_dataset
from torchvision import transforms
from models.siamese import SiameseDataset
import torch.nn.functional as F
import os
import cv2

def siamese_experiment(force_retrain=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'models/saved/siamese_model.pth'
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((45, 45)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # 检查是否存在预训练模型
    if not force_retrain and os.path.exists(model_path):
        print("加载预训练模型...")
        # 安全加载
        with torch.serialization.safe_globals([SiameseNetwork]):
            net = torch.load(
                model_path,
                weights_only=False,  # 启用安全加载
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
        criterion = TripletLoss(margin=1.0)
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
    
    # 在评估完成后添加
    print("\n生成预测可视化...")
    visualize_predictions(net, test_loader, device, base_test)
    
    return net

def evaluate_siamese(model, test_loader, device):
    model.eval()
    char_correct = 0
    char_total = 0
    captcha_correct = 0
    captcha_total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            # 将嵌套维度展开为批量维度
            char_imgs = batch[0].view(-1, 1, 45, 45).to(device)  # [batch*4, 1, 45, 45]
            pos_imgs = batch[1].view(-1, 1, 45, 45).to(device)   # [batch*4, 1, 45, 45]
            
            # 前向传播时保持正确的维度结构
            anchor = model.forward_once(char_imgs)
            positive = model.forward_once(pos_imgs)
            
            # 距离计算后的维度恢复逻辑
            distances = F.pairwise_distance(anchor, positive)
            predictions = (distances < 0.5).long()
            
            # 恢复原始验证码结构 [batch_size, 4]
            predictions = predictions.view(-1, 4)  # 将预测结果重组为每个验证码4个字符
            
            # 单字准确率计算
            char_correct += (predictions == 0).sum().item()
            char_total += predictions.numel()
            
            captcha_pred = (predictions.sum(dim=1) == 0)
            captcha_correct += captcha_pred.sum().item()
            captcha_total += captcha_pred.size(0)

    print(f'单字验证准确率: {char_correct/char_total:.4f}')
    print(f'完整验证码准确率: {captcha_correct/captcha_total:.4f}')
    return (char_correct/char_total, captcha_correct/captcha_total)

# 可视化
def visualize_predictions(model, loader, device, base_dataset, num_samples=5):
    import matplotlib.pyplot as plt
    import numpy as np
    
    model.eval()
    indices = np.random.choice(len(base_dataset), num_samples, replace=False)
    
    plt.figure(figsize=(24, 4*num_samples))
    for plot_idx, sample_idx in enumerate(indices):
        sample = base_dataset[sample_idx]
        sample_id = sample['id']
        true_label = sample['label']
        
        # 加载原始验证码图像
        original_img = cv2.imread(os.path.join(sample['captcha_path'], f"{sample_id}.jpg"))
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # 绘制原始验证码图像
        ax = plt.subplot(num_samples, 5, plot_idx*5 + 1)
        plt.imshow(original_img)
        plt.title(f"Sample ID: {sample_id}\nTrue Label: {true_label}", fontsize=10)
        plt.axis('off')

        # 绘制四个字符预测结果
        with torch.no_grad():
            inputs = [loader.dataset[sample_idx*4 + i] for i in range(4)]
            char_imgs = torch.stack([x[0] for x in inputs]).to(device)
            pos_imgs = torch.stack([x[1] for x in inputs]).to(device)
            
            anchor = model.forward_once(char_imgs)
            positive = model.forward_once(pos_imgs)
            distances = F.pairwise_distance(anchor, positive)
            predictions = (distances < 0.5).cpu().numpy()

        for char_idx in range(4):
            ax = plt.subplot(num_samples, 5, plot_idx*5 + char_idx + 2)
            
            # 获取候选字符图像
            candidate_img = inputs[char_idx][1].cpu().numpy().squeeze()
            candidate_img = (candidate_img * 0.5 + 0.5) * 255
            
            plt.imshow(candidate_img)
            
            color = 'red' if predictions[char_idx] else 'green'
            plt.title(f"Char {char_idx+1}\ndist={distances[char_idx]:.2f}",
                     color=color, fontsize=9)
            plt.axis('off')

    plt.tight_layout()
    plt.savefig(f'figures/siamese_predictions.png')
    plt.close()
    print(f"可视化结果已保存至 figures/siamese_predictions.png")
