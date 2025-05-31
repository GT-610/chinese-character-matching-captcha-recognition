import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
loss_df = pd.read_csv('results/siamese_loss.csv', header=None, names=['Epoch', 'Loss'])

# 设置绘图风格
sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))

# 绘制主曲线
plt.plot(loss_df['Epoch'], loss_df['Loss'], 
         marker='o', color='#2c7bb6', 
         linewidth=2, markersize=8,
         label='Training Loss')

# 标注关键点
for epoch in [1, 5, 10, 15, 20]:
    loss = loss_df.loc[epoch-1, 'Loss']
    plt.annotate(f'{loss:.4f}', 
                 (epoch, loss),
                 textcoords="offset points",
                 xytext=(0,10),
                 ha='center')

# 图表装饰
plt.title('Siamese Network Training Loss (20 Epochs)', fontsize=14, pad=20)
plt.xlabel('Training Epoch', fontsize=12)
plt.ylabel('Triplet Loss', fontsize=12)  # 修改损失函数名称
plt.xticks(range(1,21))
plt.ylim(0, 0.6)  # 调整y轴范围至实际数据区间
plt.legend()

# 保存输出
plt.tight_layout()
plt.savefig('figures/siamese_loss_curve.png', dpi=300)
print("结果已保存到 figures/siamese_loss_curve.png。")
plt.close()