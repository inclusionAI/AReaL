import os
import glob
import json
import pandas as pd
from PIL import Image
# ====== 你需要改的路径参数 ======
json_dir = r"C:\Users\Antoine\code\CartoMapQA\Dataset\CartoMapQA\MapNavigation\_routes"
image_root = r"C:\Users\Antoine\code\CartoMapQA\Dataset\CartoMapQA\MapNavigation\_maps"
image_ext = ".png"
out_parquet = r"C:\Users\Antoine\data\CartoMapQA\MapNavigation_dataset.parquet"
# =================================

rows = []
json_files = sorted(glob.glob(os.path.join(json_dir, "*.json")))

for i, fp in enumerate(json_files):
    with open(fp, "r", encoding="utf-8") as f:
        obj = json.load(f)

    base = os.path.splitext(os.path.basename(fp))[0]
    image_name = f"_map{base}{image_ext}"
    image_path = os.path.join(image_root, image_name)

    # 读图片为 PIL Image 对象
    image = Image.open(image_path).convert("RGB")

    obj["id"] = i
    obj["image"] = image          # 关键：图片打包进 parquet
    obj["source_file"] = os.path.basename(fp)

    rows.append(obj)

df = pd.DataFrame(rows)
df.to_parquet(out_parquet, index=False, engine="pyarrow")

print(f"Saved parquet: {out_parquet}")
print(f"Num rows: {len(df)}")
