import pandas as pd
import os
import io
from PIL import Image
from tqdm import tqdm

# --- 配置部分 ---
parquet_folder = "./" 
output_root = "data/PACS"

# PACS 标准定义
categories = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']
domains = ['art_painting', 'cartoon', 'photo', 'sketch']

def extract_parquet(file_path):
    print(f"正在读取 {file_path}...")
    df = pd.read_parquet(file_path)
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # 1. 安全获取索引并转为整数
        try:
            # 兼容处理：有些 parquet 存的是字符串 '0'，有些是整数 0
            d_idx = int(row['domain'])
            l_idx = int(row['label'])
            
            domain_name = domains[d_idx]
            category_name = categories[l_idx]
        except (ValueError, TypeError):
            # 如果 parquet 直接存的就是字符串名称（如 'art_painting'）
            domain_name = str(row['domain'])
            category_name = str(row['label'])

        # 2. 提取图像数据
        img_dict = row['image']
        img_data = img_dict['bytes']
        
        # 3. 创建目录：data/PACS/domain/category/
        save_dir = os.path.join(output_root, domain_name, category_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # 4. 还原并保存图片
        image = Image.open(io.BytesIO(img_data)).convert('RGB')
        image.save(os.path.join(save_dir, f"{_}.jpg"))

# 扫描并处理
for f in os.listdir(parquet_folder):
    if f.endswith(".parquet"):
        extract_parquet(os.path.join(parquet_folder, f))

print(f"✅ 全部转换完成！请检查 {output_root}")