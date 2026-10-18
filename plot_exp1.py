"""
plot_exp1.py - 分析一可视化
读取 experiment_1_results.csv，画散点图展示相机平均可见度与重建质量的关系
并计算皮尔逊相关系数、斯皮尔曼相关系数，以及按可见度分组统计
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 读取数据
data = np.loadtxt('experiment_1_results.csv', delimiter=',', skiprows=1)

avg_visibility = data[:, 0]  # 相机平均可见度
accuracy = data[:, 1]         # 准确度
completeness = data[:, 2]     # 完整度

# ==================== 定量分析 ====================

# 皮尔逊相关系数（线性相关）
pearson_comp, p_comp = stats.pearsonr(avg_visibility, completeness)
pearson_acc, p_acc = stats.pearsonr(avg_visibility, accuracy)

# 斯皮尔曼秩相关系数（单调相关）
spearman_comp, sp_comp = stats.spearmanr(avg_visibility, completeness)
spearman_acc, sp_acc = stats.spearmanr(avg_visibility, accuracy)

print("=" * 50)
print("  定量分析结果")
print("=" * 50)
print()
print("相机平均可见度 vs 完整度：")
print(f"  皮尔逊 r = {pearson_comp:+.4f}  (p = {p_comp:.4f})")
print(f"  斯皮尔曼 ρ = {spearman_comp:+.4f}  (p = {sp_comp:.4f})")
print()
print("相机平均可见度 vs 准确度：")
print(f"  皮尔逊 r = {pearson_acc:+.4f}  (p = {p_acc:.4f})")
print(f"  斯皮尔曼 ρ = {spearman_acc:+.4f}  (p = {sp_acc:.4f})")
print()
print("解读：r/ρ 接近 0 = 无相关，±0.3 = 弱相关，±0.5 = 中等相关，±0.7 = 强相关")
print("      p < 0.05 表示统计显著")
print("=" * 50)

# ==================== 分组统计 ====================

# 将可见度等分为 5 组
bins = 5
bin_edges = np.linspace(avg_visibility.min(), avg_visibility.max(), bins + 1)
bin_labels = [f'{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}' for i in range(bins)]

# 每组内完整度和准确度的均值和标准差
comp_means = []
comp_stds = []
acc_means = []
acc_stds = []

for i in range(bins):
    mask = (avg_visibility >= bin_edges[i]) & (avg_visibility < bin_edges[i+1])
    if i == bins - 1:
        mask = (avg_visibility >= bin_edges[i]) & (avg_visibility <= bin_edges[i+1])

    comp_means.append(np.mean(completeness[mask]))
    comp_stds.append(np.std(completeness[mask]))
    acc_means.append(np.mean(accuracy[mask]))
    acc_stds.append(np.std(accuracy[mask]))

# ==================== 可视化 ====================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

ax1 = axes[0, 0]
ax2 = axes[0, 1]
ax3 = axes[1, 0]
ax4 = axes[1, 1]

# ---- 左上：可见度 vs 完整度 ----
ax1.scatter(avg_visibility, completeness, alpha=0.6, edgecolors='black', linewidths=0.3)
ax1.set_xlabel('Camera Average Visibility', fontsize=11)
ax1.set_ylabel('Completeness', fontsize=11)
ax1.set_title('Visibility vs. Completeness', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.text(0.05, 0.95,
         f'Pearson r = {pearson_comp:+.3f}\nSpearman ρ = {spearman_comp:+.3f}',
         transform=ax1.transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ---- 右上：可见度 vs 准确度 ----
ax2.scatter(avg_visibility, accuracy, alpha=0.6, edgecolors='black', linewidths=0.3)
ax2.set_xlabel('Camera Average Visibility', fontsize=11)
ax2.set_ylabel('Accuracy', fontsize=11)
ax2.set_title('Visibility vs. Accuracy', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.text(0.05, 0.95,
         f'Pearson r = {pearson_acc:+.3f}\nSpearman ρ = {spearman_acc:+.3f}',
         transform=ax2.transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ---- 左下：完整度分组柱状图 ----
x_pos = np.arange(bins)
ax3.bar(x_pos, comp_means, yerr=comp_stds, capsize=5,
        color='coral', edgecolor='black', linewidth=0.5)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(bin_labels, rotation=30, fontsize=8)
ax3.set_xlabel('Visibility Range', fontsize=11)
ax3.set_ylabel('Mean Completeness', fontsize=11)
ax3.set_title('Completeness by Visibility Group', fontsize=12)
ax3.grid(True, alpha=0.3, axis='y')

# ---- 右下：准确度分组柱状图 ----
ax4.bar(x_pos, acc_means, yerr=acc_stds, capsize=5,
        color='steelblue', edgecolor='black', linewidth=0.5)
ax4.set_xticks(x_pos)
ax4.set_xticklabels(bin_labels, rotation=30, fontsize=8)
ax4.set_xlabel('Visibility Range', fontsize=11)
ax4.set_ylabel('Mean Accuracy', fontsize=11)
ax4.set_title('Accuracy by Visibility Group', fontsize=12)
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('experiment_1_analysis.png', dpi=150, bbox_inches='tight')
plt.show()