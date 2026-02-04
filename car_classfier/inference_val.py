"""
valフォルダの全画像を推論してCSV出力するスクリプト（クラス順を自動取得版）
出力: results_val.csv
内容: file_path, true_label, pred_label, confidence, prob_<class0>, prob_<class1>
"""

import torch
import torch.nn as nn
from torchvision import transforms, models, datasets
from pathlib import Path
from PIL import Image
import pandas as pd
from tqdm import tqdm

# ==== 設定 ====
val_dir = Path("data/val")
weight_path = Path("vehicle_model.pt")
output_csv = "results_val.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
threshold = 0.6  # needs_reviewのしきい値

# ==== クラス順を val から取得 ====
val_ds_for_map = datasets.ImageFolder(val_dir)  # フォルダ名アルファベット順で class_to_idx が作られる
class_to_idx = val_ds_for_map.class_to_idx          # 例: {'large': 0, 'small': 1}
idx_to_class = {v: k for k, v in class_to_idx.items()}
print("class_to_idx:", class_to_idx)

# ==== 推論用Transform ====
val_tfm = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225]),
])

# ==== モデル ====
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model.load_state_dict(torch.load(weight_path, map_location=device))
model = model.to(device)
model.eval()

# ==== 推論 ====
results = []
for true_label in sorted(class_to_idx.keys()):  # ['large','small'] の順になるはず
    files = sorted((val_dir / true_label).glob("*"))
    for f in tqdm(files, desc=f"Processing {true_label}"):
        try:
            img = Image.open(f).convert("RGB")
        except Exception as e:
            print(f"読み込みエラー: {f} ({e})")
            continue

        x = val_tfm(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            prob = torch.softmax(logits, dim=1)[0].cpu()

        conf, idx = prob.max(0)
        pred_name = idx_to_class[idx.item()]  # 学習時の並びに合わせた名前を使う
        label_out = pred_name if conf.item() >= threshold else "needs_review"

        # 列名がわかりやすいように確率列もクラス名で出す
        row = {
            "file_path": str(f),
            "true_label": true_label,
            "pred_label": label_out,
            "confidence": round(conf.item(), 4),
        }
        for i in range(2):
            row[f"prob_{idx_to_class[i]}"] = round(prob[i].item(), 4)
        results.append(row)

# ==== CSV出力 ====
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False, encoding="utf-8-sig")
print(f"\n推論完了。結果を {output_csv} に出力しました。")

# ==== 集計（needs_review除外で精度表示） ====
eval_df = df[df["pred_label"] != "needs_review"]
if len(eval_df) > 0:
    acc = (eval_df["true_label"] == eval_df["pred_label"]).mean()
    print(f"除外（needs_review）以外の精度: {eval_df.shape[0]- (eval_df['true_label'] != eval_df['pred_label']).sum()}/{eval_df.shape[0]} = {acc:.3f}")
else:
    print("評価対象なし。")

# ==== 簡易混同行列 ====
print("\n--- Confusion Matrix (簡易集計) ---")
print(eval_df.groupby(["true_label", "pred_label"]).size().unstack(fill_value=0))
