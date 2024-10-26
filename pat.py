import os
import requests
from bs4 import BeautifulSoup
import time

search_term = "萌奶龙"
download_path = r"D:\code\milk loong\dataset"
max_images = 100  # 设置最大爬取数量
image_count = 81

if not os.path.exists(download_path):
    os.makedirs(download_path)

url = f"https://www.google.com/search?hl=zh-CN&tbm=isch&q={search_term}"
headers = {"User-Agent": "Mozilla/5.0"}

response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, "html.parser")

for img in soup.find_all("img"):
    if image_count >= max_images:
        break

    img_url = img.get("src")
    if img_url:
        try:
            img_data = requests.get(img_url).content
            img_name = os.path.join(download_path, f"{image_count + 1}.jpg")
            with open(img_name, "wb") as img_file:
                img_file.write(img_data)
            print(f"Downloaded {img_name}")
            image_count += 1
            time.sleep(1)  # 暂停1秒
        except Exception as e:
            print(f"Could not download image from {img_url}: {e}")
