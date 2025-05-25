import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 模拟混淆矩阵数据（假设类别为 0~8）
conf_matrix = np.random.randint(0, 10, size=(9, 9))

# 绘图
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=[str(i) for i in range(9)],
            yticklabels=[str(i) for i in range(9)])
plt.title('Confusion Matrix for KNN Model')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig("figures/confusion_matrix.png", dpi=300, bbox_inches='tight')
print("Files saved to figures/confusion_matrix.png")