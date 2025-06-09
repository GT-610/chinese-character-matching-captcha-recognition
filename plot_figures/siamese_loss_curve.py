import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the loss data from CSV file
loss_df = pd.read_csv('results/siamese_loss.csv', header=None, names=['Epoch', 'Loss'])

# Set the plot style and figure size
sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))

# Plot the main curve with specified parameters
plt.plot(loss_df['Epoch'], loss_df['Loss'], 
         marker='o', color='#2c7bb6', 
         linewidth=2, markersize=8,
         label='Training Loss')

# Annotate key points on the curve
for epoch in [1, 5, 10, 15, 20]:
    loss = loss_df.loc[epoch-1, 'Loss']
    plt.annotate(f'{loss:.4f}', 
                 (epoch, loss),
                 textcoords="offset points",
                 xytext=(0,10),
                 ha='center')

# Customize the chart title, labels, and ticks
plt.title('Siamese Network Training Loss (20 Epochs)', fontsize=14, pad=20)
plt.xlabel('Training Epoch', fontsize=12)
plt.ylabel('Triplet Loss', fontsize=12)
plt.xticks(range(1,21))
plt.ylim(0, 0.6)
plt.legend()

# Save the plot to a file and close the figure
plt.tight_layout()
plt.savefig('figures/siamese_loss_curve.png', dpi=300)
print("The result has been saved to figures/siamese_loss_curve.png.")
plt.close()