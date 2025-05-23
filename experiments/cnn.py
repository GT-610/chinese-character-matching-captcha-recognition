import torch
from torch.utils.data import DataLoader
from models.cnn_classifier import CNNCharClassifier, CharDataset, train_cnn, evaluate_cnn
from data_process.load_dataset import load_dataset
from torchvision import transforms

def cnn_experiment():
    # 数据预处理：调整为150x45的输入尺寸
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((150, 45)),  # 匹配验证码图片尺寸
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # 修改为单通道灰度图像的归一化参数
    ])

    # 加载数据集
    train_dataset = CharDataset(load_dataset(data_root='data', train=True), transform=transform)
    test_dataset = CharDataset(load_dataset(data_root='data', train=False), transform=transform)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # 初始化模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CNNCharClassifier(num_classes=10, num_positions=4).to(device)  # 修改为10类和4个位置

    # 训练模型
    print("开始训练CNN模型...")
    train_cnn(model, train_loader, test_loader, epochs=10, device=device)

    # 最终评估
    final_acc = evaluate_cnn(model, test_loader, device)
    print(f"\nCNN模型最终测试准确率: {final_acc:.2%}")
