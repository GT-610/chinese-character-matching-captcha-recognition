import matplotlib.pyplot as plt
import numpy as np

# 数据准备
models = ['KNN\n(HOG)', 'CNN', 'ResNet-50\nFine-tuned', 'Siamese\nNetwork', 'Siamese+Transformer']
single_char_acc = [12.3, 75.6, 81.8, 88.0, 90.0]
sequence_acc = [0.2, 38.2, 45.6, 97.40, 98.20]

x = np.arange(len(models))  # x轴位置
width = 0.35  # 柱子宽度

# 绘图
fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, single_char_acc, width, label='Single Character Accuracy')
rects2 = ax.bar(x + width/2, sequence_acc, width, label='Sequence Accuracy')

# 添加标签和标题
ax.set_ylabel('Accuracy (%)')
ax.set_title('Performance Comparison of Different Models on Captcha Recognition')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

# 显示数值
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(rects1)
autolabel(rects2)

plt.ylim(0, 100)
plt.tight_layout()
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.savefig("figures/model_comparison_bar_chart.png", dpi=300, bbox_inches='tight')
print("Result saved to model_comparison_bar_chart.png file.")