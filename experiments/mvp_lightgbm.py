#!/usr/bin/env python3
"""MVP LightGBM — 验证统计模型能否突破规则系统 45% 上限"""
import warnings, duckdb, pandas as pd, numpy as np
warnings.filterwarnings('ignore')
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# ── 加载 ──
con = duckdb.connect()
base_m = '/home/jqs/Downloads'

files_m = [
    '0615双人对话done_qc-jiang0_de554760_qc_result.csv',
    '0615双人对话done_qc-jiang2_8f16ec23_qc_result.csv',
    '0615双人对话done_qc-jiang3_0fed46b0_qc_result.csv',
    '0615双人对话done_qc-jiang_5d8b61bd_qc_result.csv',
]
dfs = []
for f in files_m:
    path = f'{base_m}/{f}'
    df = con.execute(f"SELECT * FROM read_csv_auto('{path}', header=true, all_varchar=true) WHERE qc_result IN ('T','F')").fetchdf()
    dfs.append(df)
df = pd.concat(dfs, ignore_index=True)
con.close()

df['label'] = (df['qc_result'] == 'T').astype(int)
print(f"数据: {len(df)} 条 | T={df.label.sum()} | F={len(df)-df.label.sum()} | 通过率={df.label.mean()*100:.1f}%")

# ═══════════════ STEP 1: 特征构建 ═══════════════
print(f"\n{'='*60}\nSTEP 1: 特征构建\n{'='*60}")

# 结构
df['title_len'] = df['title'].fillna('').str.len()
df['desc_len'] = df['description'].fillna('').str.len()
df['duration_min'] = pd.to_numeric(df['duration_seconds'], errors='coerce').fillna(0) / 60
df['has_desc'] = (df['desc_len'] > 20).astype(int)

# 频道编码
ch_rate = df.groupby('channel')['label'].mean()
df['ch_score'] = df['channel'].map(ch_rate).fillna(0.5)
df['ch_freq'] = df['channel'].map(df['channel'].value_counts())

# 文本弱特征
t = df['title'].fillna('').str.lower()
for w in ['guest', 'podcast', 'interview', 'reaction', 'short',
          'talk', 'conversation', 'discussion', 'debate', 'couple',
          'episode', 'show', 'chat', 'news']:
    df[f'has_{w}'] = t.str.contains(rf'\b{w}\b').astype(int)
df['title_wc'] = t.str.split().str.len()

features = [
    'title_len', 'desc_len', 'duration_min', 'has_desc',
    'ch_score', 'ch_freq',
] + [f'has_{w}' for w in ['guest', 'podcast', 'interview', 'reaction', 'short',
                           'talk', 'conversation', 'discussion', 'debate', 'couple',
                           'episode', 'show', 'chat', 'news']] + ['title_wc']

X = df[features].copy()
y = df['label'].values

print(f"特征数: {len(features)}")

# ═══════════════ STEP 2+3: 训练 + 评估 ═══════════════
print(f"\n{'='*60}\nSTEP 2+3: 训练 LightGBM + 评估\n{'='*60}")

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = lgb.LGBMClassifier(
    n_estimators=300, learning_rate=0.05, num_leaves=31,
    min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
    random_state=42, verbose=-1,
)
model.fit(X_tr, y_tr)

pred = model.predict(X_te)
proba = model.predict_proba(X_te)[:, 1]
auc = roc_auc_score(y_te, proba)

print(f"\nAUC: {auc:.3f}")
print(f"\n{classification_report(y_te, pred, target_names=['FAIL','PASS'])}")

# ═══════════════ STEP 4: 核心验证 ═══════════════
print(f"{'='*60}\nSTEP 4: MVP 核心验证\n{'='*60}")

test_df = pd.DataFrame({'proba': proba, 'true': y_te})
base_prec = y_te.mean()

print(f"\nAUC = {auc:.3f}  {'✅ >0.70 = 突破' if auc > 0.70 else '❌ <0.70 = 未突破'}")
print(f"\nPrecision@TopK:")
for k in [10, 20, 30, 50]:
    n = max(1, int(len(test_df) * k / 100))
    top = test_df.nlargest(n, 'proba')
    prec = top['true'].mean()
    recall = top['true'].sum() / y_te.sum()
    print(f"  Top {k:>2}%: n={len(top):>3}  precision={prec*100:.1f}%  recall={recall*100:.1f}%")

print(f"\n基准 (全量):  precision={base_prec*100:.1f}%")

# ═══════════════ STEP 5: vs RULE SYSTEM ═══════════════
print(f"\n{'='*60}\nSTEP 5: 规则 vs 模型对比\n{'='*60}")

top10_prec = test_df.nlargest(max(1, int(len(test_df)*0.1)), 'proba')['true'].mean()
top20 = test_df.nlargest(max(1, int(len(test_df)*0.2)), 'proba')
top20_prec = top20['true'].mean()
top20_recall = top20['true'].sum() / y_te.sum()

print(f"""
┌─────────────────────────────────────────────┐
│  RULE SYSTEM                                │
│  precision = 45.0% (maxresdefault+1-2h)     │
│  coverage  = 11%  (140/1265)                │
│                                             │
│  LIGHTGBM                                   │
│  precision@Top20% = {top20_prec*100:.1f}%                       │
│  recall@Top20%    = {top20_recall*100:.1f}%                       │
│  AUC              = {auc:.3f}                        │
│  precision@Top10% = {top10_prec*100:.1f}%                       ││
└─────────────────────────────────────────────┘
""")

# ═══════════════ STEP 6: 判断 ═══════════════
print(f"{'='*60}\nSTEP 6: 最终判断\n{'='*60}")

if auc > 0.70 and top20_prec > 0.60:
    print(f"🟢 ML VALIDATED — LightGBM Top20%={top20_prec*100:.1f}%, 建议部署")
elif auc > 0.65 and top20_prec > 0.50:
    print(f"🟡 ML IMPROVES — Top20%={top20_prec*100:.1f}%, 建议: ML + embedding + 视觉模型")
else:
    print(f"🔴 NEED MORE FEATURES — AUC={auc:.3f}, Top20%={top20_prec*100:.1f}%")
    print("   结构特征到顶了, 需要: 文本Embedding / LLM / 视觉投票")

# ═══════════════ Feature Importance ═══════════════
print(f"\n{'='*60}\nFeature Importance (Top 10)\n{'='*60}")
imp = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
imp = imp.sort_values('importance', ascending=False).head(10)
for _, row in imp.iterrows():
    bar = '█' * int(row['importance'] / imp['importance'].max() * 40)
    print(f"  {row['feature']:<20} {row['importance']:.4f} {bar}")
