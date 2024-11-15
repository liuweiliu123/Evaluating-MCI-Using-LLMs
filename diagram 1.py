import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches

# 读取数据
data = pd.read_csv('1234567.csv', usecols=[0, 1, 2, 3])  # 读取前四列

# 假设列名未知，给列分配临时名字
data.columns = ['Color', 'Size', 'X', 'Y']

# 将 'X' 列的数据转换为数值类型，如果转换失败则设为 NaN
data['X'] = pd.to_numeric(data['X'], errors='coerce')

# 确保 Y 列也是数值型
data['Y'] = pd.to_numeric(data['Y'], errors='coerce')

# 将 'Size' 列转换为数值，以便用于散点图的大小设置
data['Size'] = pd.to_numeric(data['Size'], errors='coerce')

# 为不同的 'Color' 分配颜色
color_mapping = {'Gemma': 'red', 'Qwen': 'blue','Qwen1.5': "green",'Llama3':"yellow"}
data['Color'] = data['Color'].map(color_mapping)

# 过滤 X 列，只保留 4, 8, 12 的数据
data = data[data['X'].isin([4, 8, 12])]

# 移除任何含有 NaN 值的行，以避免绘图错误
data.dropna(subset=['X', 'Y', 'Size', 'Color'], inplace=True)

# 绘制散点图
for label, df in data.groupby('Color'):
    if label == 'red':
        model_name = 'Gemma'
    elif label == 'blue':
        model_name = 'Qwen'
    elif label == 'green':
        model_name = 'Qwen1.5'
    else:
        model_name = 'Llama3'
    plt.scatter(df['X'], df['Y'], s=df['Size'] * 150, c=label, alpha=0.5, label=model_name)

# 设置坐标轴标签
plt.xlabel('Quantization level')
plt.ylabel('Accuracy')

# 自定义横坐标刻度和标签，将 16 改为 'None'
plt.xticks([4, 8, 12], ['4', '8', 'None'])

# 设置横坐标范围
plt.xlim(3, 17)

# 创建图例
legend_handles = [
    mpatches.Patch(color='red', label='Gemma'),
    mpatches.Patch(color='blue', label='Qwen'),
    mpatches.Patch(color='green', label='Qwen1.5'),
    mpatches.Patch(color='yellow', label='Llama3')
]
plt.legend(handles=legend_handles, title='Model', bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=1)

# 调整布局，确保图例不会被剪切
plt.tight_layout()

# 显示图形
plt.show()

