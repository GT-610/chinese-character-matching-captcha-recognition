import csv
import matplotlib.pyplot as plt

def plot_validation_accuracy(data=None, csv_path=None, output_path='figures/validation_accuracy_plot.png'):
    """
    绘制验证准确率变化曲线。
    
    :param data: 验证准确率列表，格式为 [(epoch, accuracy)]。
    :param csv_path: CSV 文件路径，优先级高于 data 参数。
    :param output_path: 输出图像路径。
    """
    if csv_path:
        # 从 CSV 文件读取数据
        data = []
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # 跳过表头
            for row in reader:
                data.append((int(row[0]), float(row[1])))
    elif not data:
        raise ValueError("必须提供 data 或 csv_path 参数之一。")

    # 绘制曲线
    plt.figure(figsize=(10, 5))
    plt.plot([x[0] for x in data], [x[1] for x in data], marker='o', linestyle='-')
    plt.title('Validation Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

    print(f"验证准确率曲线已保存到 {output_path}")