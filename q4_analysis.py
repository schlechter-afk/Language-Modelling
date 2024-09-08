import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style for better aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12

models = ['5-gram LM', 'LSTM LM', 'Transformer LM']

train_average_perplexity = [183.65, 79.03, 102.24]
train_median_perplexity = [143.50, 67.50, 87.04]

test_average_perplexity = [397.12, 159.06, 137.53]
test_median_perplexity = [280.96, 119.85, 105.29]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

sns.barplot(x=models, y=train_average_perplexity, ax=axes[0, 0], hue=models, palette="Blues_d", dodge=False)
axes[0, 0].set_title('Average Train Perplexity')
axes[0, 0].set_ylabel('Average Perplexity')

sns.barplot(x=models, y=train_median_perplexity, ax=axes[0, 1], hue=models, palette="Greens_d", dodge=False)
axes[0, 1].set_title('Median Train Perplexity')
axes[0, 1].set_ylabel('Median Perplexity')

sns.barplot(x=models, y=test_average_perplexity, ax=axes[1, 0], hue=models, palette="Blues_d", dodge=False)
axes[1, 0].set_title('Average Test Perplexity')
axes[1, 0].set_ylabel('Average Perplexity')

sns.barplot(x=models, y=test_median_perplexity, ax=axes[1, 1], hue=models, palette="Greens_d", dodge=False)
axes[1, 1].set_title('Median Test Perplexity')
axes[1, 1].set_ylabel('Median Perplexity')

plt.tight_layout()
plt.savefig('perplexity.png')
plt.show()

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

plt.figure(figsize=(12, 6))

plt.plot(models, train_average_perplexity, label='Train Average', marker='o', color=colors[0], linestyle='-')
plt.plot(models, train_median_perplexity, label='Train Median', marker='o', color=colors[1], linestyle='--')
plt.plot(models, test_average_perplexity, label='Test Average', marker='o', color=colors[2], linestyle='-')
plt.plot(models, test_median_perplexity, label='Test Median', marker='o', color=colors[3], linestyle='--')

plt.xlabel('Language Models')
plt.ylabel('Perplexity')
plt.title('Performance Analysis of Language Models')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('performance_analysis.png')
plt.show()