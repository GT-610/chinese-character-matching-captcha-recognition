import torch
from torch.utils.data import DataLoader
from models.cnn_classifier import CNNCharClassifier, CharDataset, train_cnn, evaluate_cnn
from data_process.load_dataset import load_dataset
from torchvision import transforms
import os

def cnn_experiment():
    # 数据预处理：调整为150x45的输入尺寸
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((150, 45)),  # 匹配验证码图片尺寸
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集
    train_dataset = CharDataset(load_dataset(data_root='data', train=True), transform=transform)
    test_dataset = CharDataset(load_dataset(data_root='data', train=False), transform=transform)
    
    # 新增数据验证
    print(f"\n=== 数据验证 ===")
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")
    
    # 可视化第一个样本
    sample_img, sample_label = train_dataset[0]
    print(f"\n第一个训练样本:")
    print(f"图像尺寸: {sample_img.shape}")
    print(f"标签: {sample_label}")
    print(f"像素范围: [{sample_img.min():.3f}, {sample_img.max():.3f}]")

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # 初始化模型（确保使用修正后的维度）
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CNNCharClassifier(num_classes=10).to(device)  # 明确指定10个类别

    # 检查是否有预训练模型
    model_path = 'cnn_model.pth'
    if os.path.exists(model_path):
        print("加载预训练模型...")
        model.load_state_dict(torch.load(model_path))
    else:
        print("开始训练新模型...")

    # 训练模型
    train_cnn(model, train_loader, test_loader, epochs=30, device=device)

    # 保存模型
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存到 {model_path}")

    # 最终评估
    final_acc = evaluate_cnn(model, test_loader, device)
    print(f"\n验证码整体准确率: {final_acc:.2%}")