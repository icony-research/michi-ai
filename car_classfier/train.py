"""
Vehicle Classifier (Small vs Large)
-----------------------------------
- データ構成: data/train/{small,large}, data/val/{small,large}
- 不均衡対応: WeightedRandomSampler + FocalLoss
- モデル: EfficientNet-B0 転移学習
- 出力: vehicle_model.pt（最良モデル）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from PIL import Image
from pathlib import Path
import numpy as np
from sklearn.metrics import precision_recall_curve
import os

# ==== 1. 設定 ====
data_dir = Path("data")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
epochs = 15
save_path = "vehicle_model.pt"

# ==== 2. Augmentation設定 ====
strong_aug = transforms.Compose([
    transforms.Resize((288,288)),
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.ColorJitter(0.3, 0.3, 0.2, 0.05),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])
base_aug = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])
val_tfm = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

# ==== 3. Dataset ====
train_base = datasets.ImageFolder(data_dir/"train")
val_ds = datasets.ImageFolder(data_dir/"val", transform=val_tfm)

class PerClassAugDataset:
    def __init__(self, imagefolder, aug_small, aug_large):
        self.ds = imagefolder
        self.aug_small = aug_small
        self.aug_large = aug_large
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        path, y = self.ds.samples[i]
        img = Image.open(path).convert("RGB")
        x = self.aug_large(img) if y==1 else self.aug_small(img)
        return x, y

train_ds = PerClassAugDataset(train_base, base_aug, strong_aug)

# ==== 4. サンプリングによるバランス調整 ====
labels = np.array([y for _, y in train_base.samples])
class_counts = np.bincount(labels)  # [1250,163]など
class_weights = 1.0 / class_counts
sample_weights = class_weights[labels]
sampler = WeightedRandomSampler(weights=sample_weights,
                                num_samples=len(sample_weights),
                                replacement=True)

train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                          num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)

# ==== 5. FocalLoss定義 ====
class FocalLoss(nn.Module):
    def __init__(self, alpha=(1.0, 3.0), gamma=2.0):
        super().__init__()
        self.alpha = torch.tensor(alpha)
        self.gamma = gamma
    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, reduction='none')
        pt = torch.exp(-ce)
        a = self.alpha.to(logits.device)[target]
        loss = a * ((1-pt) ** self.gamma) * ce
        return loss.mean()

criterion = FocalLoss(alpha=(1.0, 3.0), gamma=2.0)

# ==== 6. モデル構築 ====
model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# ==== 7. 評価関数 ====
def evaluate():
    model.eval()
    correct, n = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            n += y.size(0)
    return correct / n

# ==== 8. 学習ループ ====
best_acc = 0.0
for epoch in range(epochs):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
    scheduler.step()
    acc = evaluate()
    print(f"[Epoch {epoch+1}/{epochs}] val_acc={acc:.4f}")
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), save_path)

print("Best val_acc:", best_acc)

# ==== 9. 閾値最適化 ====
logits_all, y_all = [], []
model.eval()
with torch.no_grad():
    for x, y in val_loader:
        x = x.to(device)
        logits = model(x)
        logits_all.append(logits.cpu())
        y_all.append(y)
logits_all = torch.cat(logits_all).numpy()
y_all = torch.cat(y_all).numpy()

probs = torch.softmax(torch.from_numpy(logits_all), dim=1).numpy()[:,1]
prec, rec, thr = precision_recall_curve(y_all, probs)
f1 = 2*prec*rec/(prec+rec+1e-9)
best_idx = np.argmax(f1)
best_thr = float(thr[best_idx]) if best_idx < len(thr) else 0.5
print(f"Best threshold for LARGE (by F1): {best_thr:.3f}")

# ==== 10. 推論関数 ====
def load_model(weight_path=save_path):
    m = models.efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    m.load_state_dict(torch.load(weight_path, map_location="cpu"))
    m.eval()
    return m

def classify(img_path, threshold=best_thr):
    m = load_model()
    img = Image.open(img_path).convert("RGB")
    x = val_tfm(img).unsqueeze(0)
    with torch.no_grad():
        logits = m(x)
        prob = torch.softmax(logits, dim=1)[0]
    p, idx = prob.max(0)
    label = ["small", "large"][idx.item()]
    if p.item() < threshold:
        return {"label":"needs_review", "confidence":float(p),
                "probs":{"small":float(prob[0]), "large":float(prob[1])}}
    return {"label":label, "confidence":float(p),
            "probs":{"small":float(prob[0]), "large":float(prob[1])}}

# ==== 11. テスト ====
# ex: result = classify("test_images/001.jpg")
# print(result)
