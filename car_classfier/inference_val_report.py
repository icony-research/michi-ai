"""
valフォルダの全画像を推論し、CSVとHTMLレポートを生成
-------------------------------------------------------
出力:
- results_val.csv
- results_val.html  ← サムネ付き/色分け/検索・フィルタUI

ポイント:
- クラス順は ImageFolder(class_to_idx) から自動取得（反転事故防止）
- needs_review しきい値は threshold で調整
"""

import torch
import torch.nn as nn
from torchvision import transforms, models, datasets
from pathlib import Path
from PIL import Image
import pandas as pd
from tqdm import tqdm
import html
import numpy as np

# ===== 設定 =====
val_dir = Path("data/val")
weight_path = Path("vehicle_model.pt")
output_csv = "results_val.csv"
output_html = "results_val.html"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
threshold = 0.6  # これ未満は needs_review にする

# ===== クラス順（学習時と一致させる） =====
val_ds_for_map = datasets.ImageFolder(val_dir)
class_to_idx = val_ds_for_map.class_to_idx          # 例: {'large':0, 'small':1}
idx_to_class = {v: k for k, v in class_to_idx.items()}
class_names_sorted = [idx_to_class[i] for i in range(len(idx_to_class))]
print("class_to_idx:", class_to_idx)

# ===== Transform =====
val_tfm = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225]),
])

# ===== モデル =====
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model.load_state_dict(torch.load(weight_path, map_location=device))
model = model.to(device)
model.eval()

# ===== 推論 =====
results = []
for true_label in sorted(class_to_idx.keys()):
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
        pred_name = idx_to_class[idx.item()]
        label_out = pred_name if conf.item() >= threshold else "needs_review"

        row = {
            "file_path": str(f),
            "true_label": true_label,
            "pred_label": label_out,
            "confidence": round(conf.item(), 4),
        }
        # 確率列をクラス名で付与
        for i in range(len(idx_to_class)):
            cname = idx_to_class[i]
            row[f"prob_{cname}"] = round(prob[i].item(), 4)
        results.append(row)

# ===== CSV出力 =====
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False, encoding="utf-8-sig")
print(f"\n推論完了。結果を {output_csv} に出力しました。")

# ===== メトリクス計算（needs_review除外） =====
eval_df = df[df["pred_label"] != "needs_review"].copy()
summary_html = ""
if len(eval_df) > 0:
    acc = (eval_df["true_label"] == eval_df["pred_label"]).mean()
    # 混同行列
    cm = (eval_df
          .groupby(["true_label", "pred_label"])
          .size()
          .unstack(fill_value=0)
          .reindex(index=class_names_sorted, columns=class_names_sorted, fill_value=0))
    # クラス別再現率/適合率
    per_class = []
    for c in class_names_sorted:
        tp = cm.loc[c, c]
        fn = cm.loc[c, :].sum() - tp
        fp = cm.loc[:, c].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        per_class.append({"class": c, "precision": prec, "recall": rec})
    pc_df = pd.DataFrame(per_class)

    # HTML化
    cm_html = cm.to_html(classes="table table-compact", border=0)
    pc_html = (pc_df.assign(precision=lambda d: d["precision"].round(3),
                            recall=lambda d: d["recall"].round(3))
                    .to_html(index=False, classes="table table-compact", border=0))
    acc_txt = f"{(len(eval_df) - (eval_df['true_label'] != eval_df['pred_label']).sum())}/{len(eval_df)} = {acc:.3f}"
    summary_html = f"""
    <div class="card">
      <div class="card-title">評価サマリ（needs_review除外）</div>
      <div>Accuracy: <b>{html.escape(acc_txt)}</b></div>
      <div class="card-subtitle">混同行列</div>
      {cm_html}
      <div class="card-subtitle">クラス別 Precision / Recall</div>
      {pc_html}
    </div>
    """
else:
    summary_html = """
    <div class="card">
      <div class="card-title">評価サマリ</div>
      <div>評価対象なし（すべて needs_review）。</div>
    </div>
    """

# ===== HTMLテーブル行生成 =====
def row_to_html(r):
    fp = html.escape(r["file_path"])
    tl = html.escape(r["true_label"])
    pl = html.escape(r["pred_label"])
    conf = f'{r["confidence"]:.3f}' if isinstance(r["confidence"], (int, float)) else html.escape(str(r["confidence"]))

    status = "review" if pl == "needs_review" else ("ok" if pl == tl else "ng")

    probs_cells = ""
    for cname in class_names_sorted:
        key = f"prob_{cname}"
        probs_cells += f'<td class="num">{r.get(key, 0):.3f}</td>'

    return f"""
    <tr data-true="{tl}" data-pred="{pl}" data-status="{status}">
      <td class="thumb"><img src="{fp}" loading="lazy" /></td>
      <td>{fp}</td>
      <td>{tl}</td>
      <td><span class="badge {status}">{pl}</span></td>
      <td class="num">{conf}</td>
      {probs_cells}
    </tr>
    """

rows_html = "\n".join(row_to_html(r) for _, r in df.iterrows())

# ===== ヘッダ列 =====
prob_headers = "".join([f"<th>prob_{html.escape(c)}</th>" for c in class_names_sorted])

# ===== HTML全体 =====
html_doc = f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>Validation Results</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Hiragino Sans", "Noto Sans JP", Arial, sans-serif; margin: 16px; color: #222; }}
  .toolbar {{ display:flex; gap:12px; flex-wrap:wrap; align-items:center; margin-bottom:12px; }}
  input[type="text"], select {{ padding:6px 8px; border:1px solid #ccc; border-radius:8px; }}
  .btn {{ padding:6px 10px; border:1px solid #ccc; border-radius:8px; background:#f7f7f7; cursor:pointer; }}
  .meta {{ margin-bottom:8px; color:#666; }}
  .card {{ border:1px solid #e5e7eb; border-radius:12px; padding:12px; margin:12px 0; background:#fff; }}
  .card-title {{ font-weight:700; margin-bottom:8px; }}
  .card-subtitle {{ margin-top:8px; margin-bottom:4px; font-weight:600; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border-bottom:1px solid #eee; padding:6px 8px; vertical-align:middle; }}
  th {{ position: sticky; top: 0; background:#fafafa; z-index:1; }}
  .table-compact th, .table-compact td {{ padding:4px 6px; }}
  .thumb img {{ width: 160px; height: auto; border-radius:8px; border:1px solid #ddd; }}
  .num {{ text-align:right; tab-size: 2; }}
  .badge {{ padding:2px 8px; border-radius:999px; font-weight:600; display:inline-block; }}
  .badge.ok {{ background:#e6f6ec; color:#127a3a; border:1px solid #b9e6c9; }}
  .badge.ng {{ background:#fde8e8; color:#b91c1c; border:1px solid #fecaca; }}
  .badge.review {{ background:#fff7ed; color:#9a3412; border:1px solid #fed7aa; }}
  .muted {{ color:#666; }}
  .footer {{ margin-top:28px; color:#666; font-size:12px; }}
</style>
</head>
<body>

<h1>Validation Results</h1>
<div class="meta">source: <code>{html.escape(str(val_dir))}</code> ・ classes: {", ".join(class_names_sorted)}</div>

<div class="toolbar">
  <label>検索: <input id="q" type="text" placeholder="ファイル名・ラベルで検索"></label>
  <label>真のラベル:
    <select id="trueSel">
      <option value="">(すべて)</option>
      {"".join([f'<option value="{html.escape(c)}">{html.escape(c)}</option>' for c in class_names_sorted])}
    </select>
  </label>
  <label>予測ステータス:
    <select id="statSel">
      <option value="">(すべて)</option>
      <option value="ok">正解</option>
      <option value="ng">不正解</option>
      <option value="review">needs_review</option>
    </select>
  </label>
  <button class="btn" id="resetBtn">リセット</button>
  <a class="btn" href="{html.escape(output_csv)}" download>CSVをダウンロード</a>
</div>

{summary_html}

<table id="tbl">
  <thead>
    <tr>
      <th>image</th>
      <th>file_path</th>
      <th>true_label</th>
      <th>pred_label</th>
      <th>confidence</th>
      {prob_headers}
    </tr>
  </thead>
  <tbody>
    {rows_html}
  </tbody>
</table>

<div class="footer">
  threshold = {threshold} （これ未満は <span class="badge review">needs_review</span>）
</div>

<script>
  const q = document.getElementById('q');
  const trueSel = document.getElementById('trueSel');
  const statSel = document.getElementById('statSel');
  const resetBtn = document.getElementById('resetBtn');
  const rows = Array.from(document.querySelectorAll('#tbl tbody tr'));

  function applyFilter() {{
    const qv = q.value.toLowerCase();
    const tv = trueSel.value;
    const sv = statSel.value;
    rows.forEach(tr => {{
      const file = tr.children[1].textContent.toLowerCase();
      const trueL = tr.dataset.true;
      const stat = tr.dataset.status;
      let ok = true;
      if (qv && !file.includes(qv)) ok = false;
      if (tv && trueL !== tv) ok = false;
      if (sv && stat !== sv) ok = false;
      tr.style.display = ok ? '' : 'none';
    }});
  }}

  [q, trueSel, statSel].forEach(el => el.addEventListener('input', applyFilter));
  resetBtn.addEventListener('click', () => {{
    q.value = '';
    trueSel.value = '';
    statSel.value = '';
    applyFilter();
  }});

  // シンプルな列ソート（数値列は右寄せクラス num を判定）
  document.querySelectorAll('#tbl thead th').forEach((th, idx) => {{
    th.style.cursor = 'pointer';
    th.addEventListener('click', () => {{
      const tbody = document.querySelector('#tbl tbody');
      const arr = Array.from(tbody.rows);
      const isNum = th.classList.contains('num') || idx >= 4; // confidence以降は数値想定
      const asc = th.dataset.asc !== 'true';
      arr.sort((a,b) => {{
        let va = a.cells[idx].textContent.trim();
        let vb = b.cells[idx].textContent.trim();
        if (isNum) {{ va = parseFloat(va)||0; vb = parseFloat(vb)||0; }}
        return asc ? (va > vb ? 1 : va < vb ? -1 : 0) : (va < vb ? 1 : va > vb ? -1 : 0);
      }});
      th.dataset.asc = asc ? 'true' : 'false';
      arr.forEach(r => tbody.appendChild(r));
    }});
  }});
</script>

</body>
</html>
"""

# ===== HTML出力 =====
with open(output_html, "w", encoding="utf-8") as f:
    f.write(html_doc)

print(f"HTMLレポートを {output_html} に出力しました。ブラウザで開いてください。")
