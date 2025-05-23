import torch
from torch.utils.data import DataLoader
from models.cnn_classifier import CNNCharClassifier, CharDataset, train_cnn, evaluate_cnn
from data_process.load_dataset import load_dataset
from torchvision import transforms

def cnn_experiment():
    # 数据预处理（保持验证码整体结构）
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),  # 保持原始比例填充
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集
    train_dataset = CharDataset(load_dataset(data_root='data', train=True), transform=transform)
    test_dataset = CharDataset(load_dataset(data_root='data', train=False), transform=transform)
    
    # 数据验证
    print(f"\n=== 数据验证 ===")
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")
    
    # 检查第一个样本
    sample_img, sample_label = train_dataset[0]
    print(f"\n第一个训练样本:")
    print(f"图像尺寸: {sample_img.shape}")
    print(f"标签序列: {sample_label.tolist()}")
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # 初始化模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CNNCharClassifier(num_classes=9).to(device)

    # 训练模型
    print("开始训练验证码识别模型...")
    train_cnn(model, train_loader, test_loader, epochs=20, device=device)

    # 最终评估
    final_acc = evaluate_cnn(model, test_loader, device)
    print(f"\n验证码整体准确率: {final_acc:.2%}")