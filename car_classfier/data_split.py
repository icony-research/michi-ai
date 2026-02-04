from sklearn.model_selection import train_test_split
from pathlib import Path
import shutil

# 元データ（すべての画像が small / large の2フォルダに入っている前提）
base_dir = Path("data_raw")
train_dir = Path("data/train")
val_dir = Path("data/val")

for d in [train_dir, val_dir]:
    (d/"small").mkdir(parents=True, exist_ok=True)
    (d/"large").mkdir(parents=True, exist_ok=True)

for cls in ["small", "large"]:
    files = list((base_dir/cls).glob("*"))
    train_files, val_files = train_test_split(files, test_size=0.2, random_state=42)
    for f in train_files:
        shutil.copy(f, train_dir/cls/f.name)
    for f in val_files:
        shutil.copy(f, val_dir/cls/f.name)

print("Done. train/val ディレクトリに分割完了。")
