import torch
from torch.utils.data import DataLoader
from models.cnn_classifier import CNNCharClassifier, CharDataset, train_cnn, evaluate_cnn
from data_process.load_dataset import load_dataset
from torchvision import transforms
import os
import csv
import matplotlib.pyplot as plt

def cnn_experiment():
    # 数据预处理：调整为150x45（验证码大小）的输入尺寸
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((150, 45)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # 加载数据集
    train_dataset = CharDataset(load_dataset(data_root='data', train=True), transform=transform)
    test_dataset = CharDataset(load_dataset(data_root='data', train=False), transform=transform)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # 初始化模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CNNCharClassifier(num_classes=10, num_positions=4).to(device)

    # 训练模型
    print("开始训练CNN模型...")
    val_accs = []  # 用于存储验证准确率
    for epoch in range(10):
        train_cnn(model, train_loader, test_loader, epochs=1, device=device)
        val_acc = evaluate_cnn(model, test_loader, device)
        val_accs.append(val_acc)
        print(f'Epoch {epoch+1}/10 | Val Acc: {val_acc:.2%}')

    # 最终评估
    final_acc = evaluate_cnn(model, test_loader, device)
    print(f"\nCNN模型最终测试准确率: {final_acc:.2%}")

    # 保存结果到文件
    save_results(val_accs, final_acc)

def save_results(val_accs, final_acc):
    os.makedirs('results', exist_ok=True)

    # 保存验证准确率到 CSV 文件
    with open('results/validation_accuracy.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Validation Accuracy'])
        for i, acc in enumerate(val_accs):
            writer.writerow([i+1, acc])

    # 保存最终准确率到 CSV 文件
    with open('results/final_accuracy.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Final Accuracy'])
        writer.writerow([final_acc])

    # 绘制验证准确率变化曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 11), val_accs, marker='o', linestyle='-')
    plt.title('Validation Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig('figures/validation_accuracy_plot.png')
    plt.close()

    # 绘制各位置准确率柱状图
    position_correct = [0]*4
    for i in range(4):
        position_correct[i] += (preds[:, i] == labels[:, i]).sum().item()
    position_accs = [acc / len(test_loader.dataset) for acc in position_correct]

    plt.figure(figsize=(10, 5))
    plt.bar(range(1, 5), position_accs)
    plt.title('Accuracy by Position')
    plt.xlabel('Position')
    plt.ylabel('Accuracy')
    plt.xticks(range(1, 5))
    plt.ylim(0, 1)
    plt.grid(True)
    plt.savefig('figures/position_accuracy_plot.png')
    plt.close()

    print("结果已保存到 results 文件夹中，绘图已保存到 figures 文件夹中。")