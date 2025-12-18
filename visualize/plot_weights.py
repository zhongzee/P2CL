import json
import matplotlib.pyplot as plt
import numpy as np

# 加载保存的权重数据
with open('/root/BCD_PDA/class_weights.json', 'r') as f:
    class_weights_dict = json.load(f)

# 定义迭代次数列表
iterations = ['Iter_0', 'Iter_500', 'Iter_5000']

# 设置绘图
fig, axes = plt.subplots(1, len(iterations), figsize=(15, 5), sharey=True)

# 共享类和私有类的索引
shared_classes = range(10)
private_classes = range(10, 31)

for i, iter_key in enumerate(iterations):
    weights = np.array(class_weights_dict[iter_key])

    # 共享类权重
    axes[i].bar(shared_classes, weights[shared_classes], label='Shared Class', color='blue')
    # 私有类权重
    axes[i].bar(private_classes, weights[private_classes], label='Private Class', color='orange')

    axes[i].set_title(f'Iter {iter_key.split("_")[1]}')
    axes[i].axhline(y=0.1, color='r', linestyle='--')  # 添加红色参考线
    axes[i].set_xlabel('Class Index')
    if i == 0:
        axes[i].set_ylabel('Average Weight')

# 添加图例
axes[0].legend(loc='upper right')

# 调整每个子图的间距
plt.tight_layout()

# 保存整个图集为图片文件
plt.savefig('/root/BCD_PDA/class_weights_distribution.png')

# 显示图表
plt.show()
