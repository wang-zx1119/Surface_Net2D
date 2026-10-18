"""
generate_colorbar.py - 生成纹理丰富度颜色条展示图
每个色相范围对应一条细细的横条，从左到右渐变，上下排列
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

# 六个色相范围，从下到上排列（从单一到丰富）
hue_ranges = [0.01, 0.05, 0.10, 0.20, 0.40, 0.80]
labels = ['0.01', '0.05', '0.10', '0.20', '0.40', '0.80']

fig, axes = plt.subplots(6, 1, figsize=(10, 4))

for i, (hue_range, label) in enumerate(zip(hue_ranges, labels)):
    ax = axes[i]

    # 生成水平渐变颜色条（从左到右色相从0变化到hue_range）
    t = np.linspace(0, 1, 500)
    hsv = np.zeros((500, 3))
    hsv[:, 0] = hue_range * t
    hsv[:, 1] = 1.0
    hsv[:, 2] = 1.0
    rgb_gradient = hsv_to_rgb(hsv)

    # 绘制细横条
    rgb_image = rgb_gradient.reshape(1, 500, 3)
    ax.imshow(rgb_image, aspect='auto')

    # 在左侧标注色相范围
    ax.set_ylabel(f'{label}', fontsize=10, rotation=0, labelpad=25,
                  verticalalignment='center')
    ax.set_yticks([])
    ax.set_xticks([])

    # 细边框
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('gray')
        spine.set_linewidth(0.5)

axes[0].set_title('Texture Richness Gradient (Hue Range)', fontsize=12, fontweight='bold')
axes[-1].set_xlabel('Curve Parameter t (0 → 1)', fontsize=10)

plt.tight_layout(pad=0.5)
plt.savefig('texture_gradient.png', dpi=150, bbox_inches='tight')
plt.show()

print("图片已保存为 texture_gradient.png")
