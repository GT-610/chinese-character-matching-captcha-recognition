from data_process.load_dataset import load_dataset
from data_process.show_samples import show_samples

# 1. 读取整个数据集
dataset = load_dataset(data_root='data')

# 2. 随机展示5个样本
show_samples(dataset, num_samples=5)