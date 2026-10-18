"""
plot_exp2.py - 分析二可视化
读取 experiment_2_results.csv，画折线图展示纹理丰富度与重建质量的关系
"""

import numpy as np
import matplotlib.pyplot as plt

# 读取数据
data = np.loadtxt('experiment_2_results.csv', delimiter=',', skiprows=1)

hue_range = data[:, 0]      # 色相范围
mean_accuracy = data[:, 1]   # 平均准确度
mean_completeness = data[:, 2]  # 平均完整度

# ==================== 定量分析 ====================

from scipy import stats

pearson_acc, p_acc = stats.pearsonr(hue_range, mean_accuracy)
pearson_comp, p_comp = stats.pearsonr(hue_range, mean_completeness)
spearman_acc, sp_acc = stats.spearmanr(hue_range, mean_accuracy)
spearman_comp, sp_comp = stats.spearmanr(hue_range, mean_completeness)

print("=" * 50)
print("  定量分析结果")
print("=" * 50)
print()
print("色相范围 vs 准确度：")
print(f"  皮尔逊 r = {pearson_acc:+.4f}  (p = {p_acc:.4f})")
print(f"  斯皮尔曼 ρ = {spearman_acc:+.4f}  (p = {sp_acc:.4f})")
print()
print("色相范围 vs 完整度：")
print(f"  皮尔逊 r = {pearson_comp:+.4f}  (p = {p_comp:.4f})")
print(f"  斯皮尔曼 ρ = {spearman_comp:+.4f}  (p = {sp_comp:.4f})")
print()

# ==================== 可视化 ====================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# ---- 左图：色相范围 vs 准确度 ----
ax1.plot(hue_range, mean_accuracy, 'o-', color='steelblue', linewidth=2,
         markersize=8, markerfacecolor='white', markeredgewidth=2)
ax1.set_xlabel('Hue Range (Texture Richness)', fontsize=12)
ax1.set_ylabel('Mean Accuracy', fontsize=12)
ax1.set_title('Texture Richness vs. Accuracy', fontsize=13)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 0.85)
ax1.text(0.55, 0.95,
         f'Pearson r = {pearson_acc:+.3f}\nSpearman ρ = {spearman_acc:+.3f}',
         transform=ax1.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6))

# 标注关键转折点
ax1.annotate(f'Sharp drop\n(hue=0.10)', xy=(0.10, 0.0967),
             xytext=(0.25, 0.18), fontsize=9,
             arrowprops=dict(arrowstyle='->', color='gray'),
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ---- 右图：色相范围 vs 完整度 ----
ax2.plot(hue_range, mean_completeness, 'o-', color='coral', linewidth=2,
         markersize=8, markerfacecolor='white', markeredgewidth=2)
ax2.set_xlabel('Hue Range (Texture Richness)', fontsize=12)
ax2.set_ylabel('Mean Completeness', fontsize=12)
ax2.set_title('Texture Richness vs. Completeness', fontsize=13)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 0.85)
ax2.text(0.55, 0.95,
         f'Pearson r = {pearson_comp:+.3f}\nSpearman ρ = {spearman_comp:+.3f}',
         transform=ax2.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.6))

plt.tight_layout()
plt.savefig('experiment_2_analysis.png', dpi=150, bbox_inches='tight')
plt.show()