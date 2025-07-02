import matplotlib.pyplot as plt
import numpy as np

# 数据设置
sizes = [92, 8]

# 颜色设置
# rgba(0,78,125,255) 转换为matplotlib格式
colors = [(0/255, 78/255, 125/255, 1.0), 'white']

# 创建饼状图
fig, ax = plt.subplots(figsize=(8, 8))

# 绘制饼状图（不显示标签和百分比）
wedges, texts = ax.pie(sizes, 
                       colors=colors, 
                       startangle=90)

# 在饼状图中心添加92%文字，使用最大字体
ax.text(0, 0, '92%', fontsize=72, fontweight='bold', 
        ha='center', va='center', color='white')

# 确保饼状图是圆形
ax.axis('equal')

# 显示图表
plt.tight_layout()
plt.show()

# 保存图片
plt.savefig('pie_chart.png', dpi=300, bbox_inches='tight')
print("饼状图已生成并保存为 pie_chart.png")
