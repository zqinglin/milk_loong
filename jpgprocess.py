from PIL import Image
import os


def resize_images(input_folder, output_folder, size=(128, 128)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            img = Image.open(os.path.join(input_folder, filename))
            img = img.convert('RGB')  # 转换为 RGB 模式
            img = img.resize(size)
            img.save(os.path.join(output_folder, filename))


resize_images(r'D:\code\milk loong\images', r'D:\code\milk loong\imagesp', size=(128, 128))
