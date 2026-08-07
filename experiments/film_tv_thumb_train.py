#!/usr/bin/env python3
"""
film_tv 缩略图分类器 — MobileNetV3-Small

用 vision QC 标注（qc_thumb_result=T/F）训练缩略图二分类器：
  T = 影视海报/封面，F = 非目标封面

用法:
  python3 experiments/film_tv_thumb_train.py          # 训练 + 评估
  python3 experiments/film_tv_thumb_train.py --test   # 仅评估已保存模型
"""

import sys, os, glob, argparse, time, pickle, random
from pathlib import Path

import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

_project_root = Path(__file__).resolve().parent.parent
THUMB_CACHE = _project_root / "qc_thumb_cache"
MODEL_DIR = _project_root / "models"
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_EPOCHS = 10
LR = 0.001
IMG_SIZE = 224

# ── 数据源 ──
THUMB_QC_FILES = [
    "data/runs/film_tv/005_clean/影视剧-播放列表_771f94e0_records/run05/影视剧-播放列表_771f94e0_records_run01_keep_high_clean_0722_thumb_qc.csv",
    "data/runs/film_tv/005_clean/影视剧-频道_0316a650_records/run05/影视剧-频道_0316a650_records_clean_0722_thumb_qc.csv",
    "data/runs/film_tv/005_clean/影视剧核心片段批量_4292e310_records/run01/影视剧核心片段批量_4292e310_records_clean_0722_thumb_qc.csv",
    "data/runs/film_tv/005_clean/影视剧核心片段批量-无负面_096a9099_records/run01/影视剧核心片段批量-无负面_096a9099_records_clean_0722_thumb_qc.csv",
    "data/runs/film_tv/005_clean/影视剧_31768118_records/run01/影视剧_31768118_records_run01_keep_thumb_qc.csv",
]


# ══════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════

class ThumbDataset(Dataset):
    """从 QC 标注 + 本地缓存加载缩略图。"""

    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self._cache_miss = 0

    def _find_thumbnail(self, video_id: str) -> str | None:
        for suffix in ["maxresdefault", "hqdefault", "mqdefault", "sddefault", "0"]:
            path = THUMB_CACHE / f"{video_id}_{suffix}.jpg"
            if path.exists() and path.stat().st_size >= 1500:
                return str(path)
        return None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        vid = row["video_id"]
        label = 1 if row["qc_thumb_result"] == "T" else 0

        img_path = self._find_thumbnail(vid)
        if img_path is None:
            # 缓存未命中，返回黑图
            self._cache_miss += 1
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))
        else:
            img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.float32)

    @property
    def cache_miss(self):
        return self._cache_miss


# ══════════════════════════════════════════════════════════════
# 数据加载
# ══════════════════════════════════════════════════════════════

def load_balanced_data(max_f_samples: int = 30000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """加载所有 vision QC 数据，平衡 T/F 采样，返回 train/val DataFrame。"""
    print("加载 vision QC 数据...")
    frames = []
    for f in THUMB_QC_FILES:
        path = _project_root / f
        if not path.exists():
            print(f"  [SKIP] 文件不存在: {f}")
            continue
        df = pd.read_csv(str(path))
        labeled = df[df["qc_thumb_result"].isin(["T", "F"])].copy()
        labeled["_source"] = f.split("/")[-2]
        frames.append(labeled)
        print(f"  {f.split('/')[-1][:50]}... T={len(labeled[labeled.qc_thumb_result=='T']):,}  F={len(labeled[labeled.qc_thumb_result=='F']):,}")

    df_all = pd.concat(frames, ignore_index=True)
    # 去重 video_id
    df_all = df_all.drop_duplicates("video_id", keep="first")

    t_df = df_all[df_all["qc_thumb_result"] == "T"]
    f_df = df_all[df_all["qc_thumb_result"] == "F"]

    # 平衡采样
    n_f = min(len(f_df), max_f_samples)
    f_sampled = f_df.sample(n=n_f, random_state=RANDOM_SEED)

    print(f"\n训练集: T={len(t_df):,}  F={n_f:,}  (F 采样自 {len(f_df):,})")

    balanced = pd.concat([t_df, f_sampled], ignore_index=True)

    train_df, val_df = train_test_split(
        balanced, test_size=0.2, stratify=balanced["qc_thumb_result"],
        random_state=RANDOM_SEED,
    )
    print(f"Train/Val: {len(train_df):,} / {len(val_df):,}")
    return train_df, val_df


# ══════════════════════════════════════════════════════════════
# 模型
# ══════════════════════════════════════════════════════════════

def build_model() -> nn.Module:
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    # 替换分类头
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Identity()  # 移除最后的 Linear
    # 添加自定义分类头
    model.classifier = nn.Sequential(
        model.classifier,  # 前面的层（含 dropout）
        nn.Linear(in_features, 1),
        nn.Sigmoid(),
    )
    return model.to(DEVICE)


# ══════════════════════════════════════════════════════════════
# 训练
# ══════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(imgs)
        preds = (outputs > 0.5).float()
        correct += (preds == labels).sum().item()
        total += len(imgs)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_labels, all_probs = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs).squeeze()
        loss = criterion(outputs, labels)
        total_loss += loss.item() * len(imgs)
        preds = (outputs > 0.5).float()
        correct += (preds == labels).sum().item()
        total += len(imgs)
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(outputs.cpu().tolist())
    auc = roc_auc_score(all_labels, all_probs)
    ap = average_precision_score(all_labels, all_probs)
    return total_loss / total, correct / total, auc, ap


def train():
    print(f"Device: {DEVICE}  (GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB)" if DEVICE.type == "cuda" else f"Device: {DEVICE}")

    # ── 数据 ──
    train_df, val_df = load_balanced_data(max_f_samples=30000)

    transform_train = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_val = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = ThumbDataset(train_df, transform_train)
    val_ds = ThumbDataset(val_df, transform_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print(f"缓存未命中: train={train_ds.cache_miss} ({train_ds.cache_miss/len(train_ds)*100:.1f}%)  val={val_ds.cache_miss} ({val_ds.cache_miss/len(val_ds)*100:.1f}%)")

    # ── 模型 ──
    model = build_model()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    # ── 训练循环 ──
    best_auc = 0
    for epoch in range(NUM_EPOCHS):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, val_auc, val_ap = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS}  "
              f"train_loss={train_loss:.4f}  val_acc={val_acc:.3f}  val_auc={val_auc:.4f}  val_ap={val_ap:.4f}  "
              f"lr={optimizer.param_groups[0]['lr']:.1e}  [{elapsed:.0f}s]")
        if val_auc > best_auc:
            best_auc = val_auc
            MODEL_DIR.mkdir(exist_ok=True)
            torch.save(model.state_dict(), MODEL_DIR / "film_tv_thumb_clf.pth")
            print(f"  → 模型已保存 (AUC={best_auc:.4f})")

    print(f"\n最佳 AUC: {best_auc:.4f}")

    # ── 最终评估 ──
    model.load_state_dict(torch.load(MODEL_DIR / "film_tv_thumb_clf.pth"))
    _, _, final_auc, final_ap = evaluate(model, val_loader, criterion)
    print(f"最终 Val AUC: {final_auc:.4f}  AP: {final_ap:.4f}")


# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    train()


if __name__ == "__main__":
    main()
