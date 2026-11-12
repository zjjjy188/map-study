import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import font_manager

# 设置中文字体（Windows）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("四种视觉任务对比", fontsize=20, fontweight='bold')

# ============ 1. 分类 ============
ax = axes[0, 0]
ax.set_title("① 分类 Classification", fontsize=14, fontweight='bold')
ax.add_patch(patches.Rectangle((0.1, 0.2), 0.8, 0.6, fill=True,
                                facecolor='#FFE4B5', edgecolor='black', lw=2))
ax.text(0.5, 0.5, "🐱", fontsize=60, ha='center', va='center')
ax.text(0.5, 0.1, '输出："猫"', fontsize=14, ha='center',
        bbox=dict(boxstyle='round', facecolor='#FFD700'))
ax.text(0.5, 0.9, "整图一个标签", fontsize=11, ha='center', style='italic')
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

# ============ 2. 检测 ============
ax = axes[0, 1]
ax.set_title("② 检测 Detection", fontsize=14, fontweight='bold')
ax.add_patch(patches.Rectangle((0.1, 0.2), 0.8, 0.6, fill=True,
                                facecolor='#FFE4B5', edgecolor='black', lw=2))
ax.add_patch(patches.Rectangle((0.2, 0.35), 0.25, 0.35, fill=False,
                                edgecolor='red', lw=3))
ax.text(0.325, 0.52, "🐱", fontsize=30, ha='center', va='center')
ax.add_patch(patches.Rectangle((0.55, 0.3), 0.25, 0.4, fill=False,
                                edgecolor='blue', lw=3))
ax.text(0.675, 0.5, "🐶", fontsize=30, ha='center', va='center')
ax.text(0.5, 0.1, '输出：框 + 类别', fontsize=14, ha='center',
        bbox=dict(boxstyle='round', facecolor='#FFD700'))
ax.text(0.5, 0.9, "框出位置和类别", fontsize=11, ha='center', style='italic')
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

# ============ 3. 分割 ============
ax = axes[1, 0]
ax.set_title("③ 分割 Segmentation", fontsize=14, fontweight='bold')
ax.add_patch(patches.Rectangle((0.1, 0.2), 0.8, 0.6, fill=True,
                                facecolor='#FFE4B5', edgecolor='black', lw=2))
ax.add_patch(patches.Ellipse((0.35, 0.5), 0.3, 0.4, fill=True,
                              facecolor='red', alpha=0.5, edgecolor='red', lw=2))
ax.text(0.35, 0.5, "🐱", fontsize=30, ha='center', va='center')
ax.add_patch(patches.Ellipse((0.7, 0.45), 0.25, 0.35, fill=True,
                              facecolor='blue', alpha=0.5, edgecolor='blue', lw=2))
ax.text(0.7, 0.45, "🐶", fontsize=25, ha='center', va='center')
ax.text(0.5, 0.1, '输出：像素级掩码', fontsize=14, ha='center',
        bbox=dict(boxstyle='round', facecolor='#FFD700'))
ax.text(0.5, 0.9, "每个像素都有类别", fontsize=11, ha='center', style='italic')
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

# ============ 4. OCR ============
ax = axes[1, 1]
ax.set_title("④ OCR 文字识别", fontsize=14, fontweight='bold')
ax.add_patch(patches.Rectangle((0.1, 0.2), 0.8, 0.6, fill=True,
                                facecolor='#FFE4B5', edgecolor='black', lw=2))
ax.text(0.3, 0.6, "创新路", fontsize=20, ha='center', va='center')
ax.text(0.65, 0.45, "北行涮串", fontsize=14, ha='center', va='center')
ax.add_patch(patches.Rectangle((0.18, 0.52), 0.25, 0.16, fill=False,
                                edgecolor='green', lw=3))
ax.add_patch(patches.Rectangle((0.52, 0.38), 0.26, 0.14, fill=False,
                                edgecolor='green', lw=3))
ax.text(0.5, 0.1, '输出：文字 + 坐标', fontsize=14, ha='center',
        bbox=dict(boxstyle='round', facecolor='#FFD700'))
ax.text(0.5, 0.9, "检测+识别文字", fontsize=11, ha='center', style='italic')
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

plt.tight_layout()
plt.savefig("day7_tasks_compare.png", dpi=120, bbox_inches='tight')
print("✅ 已保存 day7_tasks_compare.png")
plt.show()