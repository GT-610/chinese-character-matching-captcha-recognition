from data_process.load_dataset import load_dataset
# from data_process.show_samples import show_samples
from feature_extract.show_samples import show_samples, show_features_visualization
from feature_extract.feature_analysis import analyze_single_char_features

# 1. 读取整个数据集
dataset = load_dataset(data_root='data')

# 2. 随机展示5个样本
# show_samples(dataset, num_samples=5)

# 3. 特征分析 - 可视化特征分布
print("Analyzing feature distribution...")
single_char_features, single_char_labels = analyze_single_char_features(dataset)

# 4. 特征分析 - 可视化具体特征
print("Visualizing HOG features...")
# 展示单个字符的特征提取效果
show_features_visualization(dataset[0]['single_char_paths'][0])

# 5. 随机展示样本及其特征分析
print("Showing sample visualizations...")
show_samples(dataset, num_samples=3)