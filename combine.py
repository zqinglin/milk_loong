import os
import shutil
import pandas as pd

# 设置路径
milk_images_path = r'D:\code\milk loong\datasetp'  # 奶龙图片路径
non_milk_images_path = r'D:\code\milk loong\imagesp'  # 非奶龙图片路径
output_images_path = r'D:\code\milk loong\combined_images'  # 合并后图片存储路径

# 创建输出目录
os.makedirs(output_images_path, exist_ok=True)

# 读取奶龙数据和非奶龙数据
milk_data = pd.read_csv('annotations1.csv')  # 奶龙数据
non_milk_data = pd.read_csv('annotations0.csv')  # 非奶龙数据

combined_data = []

# 处理奶龙图片
for index, row in milk_data.iterrows():
    original_filename = row['filename']
    src_file = os.path.join(milk_images_path, original_filename)
    if os.path.exists(src_file):
        new_filename = f'milk_{index}_{original_filename}'  # 加前缀和索引
        combined_data.append({'filename': new_filename, 'label': '1'})  # 奶龙的标签设为 1 需要是字符串
        shutil.copy(src_file, os.path.join(output_images_path, new_filename))
    else:
        print(f"File not found: {src_file}")  # 输出未找到的文件

# 处理非奶龙图片
for index, row in non_milk_data.iterrows():
    original_filename = row['filename']
    src_file = os.path.join(non_milk_images_path, original_filename)
    if os.path.exists(src_file):
        new_filename = f'non_milk_{index}_{original_filename}'  # 加前缀和索引
        combined_data.append({'filename': new_filename, 'label': '0'})  # 非奶龙的标签设为 0
        shutil.copy(src_file, os.path.join(output_images_path, new_filename))
    else:
        print(f"File not found: {src_file}")  # 输出未找到的文件

# 转换为 DataFrame 并保存
combined_df = pd.DataFrame(combined_data)
combined_df.to_csv(r'D:\code\milk loong\combined_data.csv', index=False)

print("图片和数据集合并完成！")
