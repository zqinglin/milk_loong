import os
import pandas as pd

# 输入图片文件夹路径
input_folder = r'D:\code\milk loong\imagesp'
# 创建一个空的列表用于存储文件名和标签
data = []

# 遍历文件夹中的每个文件
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # 假设所有图片都是奶龙
        data.append([filename, 'not milk loong'])

# 创建DataFrame并保存为CSV
df = pd.DataFrame(data, columns=['filename', 'label'])
df.to_csv('annotations0.csv', index=False, encoding='utf-8')
