#!/usr/bin/env python3
"""
pass_filter_ml.py — LightGBM 全量打分 + 分层抽样 + 人工验证样本导出

流程:
  1. 用人工标注数据训练无泄漏 LightGBM (5-fold CV)
  2. 对 194K pass 全量打分
  3. 按 score 分层切桶 + 抽样
  4. 导出 manual_review.csv 供人工质检

用法:
  conda run -n data_cleaning python pass_filter_ml.py
"""

import warnings, duckdb, pandas as pd, numpy as np, re, os, csv
warnings.filterwarnings('ignore')
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# ══════════════════════════════════════
# CONFIG
# ══════════════════════════════════════
BASE_M = '/home/jqs/Downloads'
BASE_AI = '/home/jqs/clean_DATASET/data/runs/tow-person/002_audit'
ALL_PASS_PATH = f'{BASE_AI}/all_four_pass.csv'
OUT_DIR = f'{BASE_AI}/ml_filter_output'
os.makedirs(OUT_DIR, exist_ok=True)

SAMPLE_SIZES = {'top10pct': 100, 'top20pct': 100, 'mid60pct': 100, 'bottom20pct': 100}
RANDOM_SEED = 42

# ══════════════════════════════════════
# 特征构建函数
# ══════════════════════════════════════
def build_features(df):
    """构建无泄漏特征 (不依赖人工标签)"""
    f = pd.DataFrame(index=df.index)

    # 结构特征
    f['title_len'] = df['title'].fillna('').str.len()
    f['desc_len'] = df['description'].fillna('').str.len()
    f['duration_min'] = pd.to_numeric(df['duration_seconds'], errors='coerce').fillna(0) / 60
    f['has_desc'] = (f['desc_len'] > 20).astype(int)

    # 缩略图类型
    qr = df['qc_result'].fillna('')
    f['thumb_maxres'] = qr.apply(lambda x: 1 if 'maxresdefault' in str(x) else 0)
    f['thumb_hq'] = qr.apply(lambda x: 1 if 'hqdefault' in str(x) else 0)

    # 文本弱特征
    t = df['title'].fillna('').str.lower()
    for w in ['guest', 'podcast', 'interview', 'reaction', 'short',
              'talk', 'conversation', 'discussion', 'debate', 'couple',
              'episode', 'show', 'chat', 'news', 'love', 'call',
              'music', 'gaming', 'official', 'vlog']:
        f[f'has_{w}'] = t.str.contains(rf'\b{w}\b').astype(int)
    f['title_wc'] = t.str.split().str.len()

    return f

FEATURE_NAMES = [
    'title_len', 'desc_len', 'duration_min', 'has_desc',
    'thumb_maxres', 'thumb_hq',
] + [f'has_{w}' for w in ['guest', 'podcast', 'interview', 'reaction', 'short',
                           'talk', 'conversation', 'discussion', 'debate', 'couple',
                           'episode', 'show', 'chat', 'news', 'love', 'call',
                           'music', 'gaming', 'official', 'vlog']] + ['title_wc']

# ══════════════════════════════════════
# Step 1: 加载人工标注数据 + 无泄漏训练
# ══════════════════════════════════════
print("=" * 60)
print("Step 1: 无泄漏 LightGBM 训练 (5-fold CV)")
print("=" * 60)

# 加载人工标注
con = duckdb.connect()
files_m = [
    '0615双人对话done_qc-jiang0_de554760_qc_result.csv',
    '0615双人对话done_qc-jiang2_8f16ec23_qc_result.csv',
    '0615双人对话done_qc-jiang3_0fed46b0_qc_result.csv',
    '0615双人对话done_qc-jiang_5d8b61bd_qc_result.csv',
]
dfs = []
for f in files_m:
    path = f'{BASE_M}/{f}'
    df = con.execute(f"SELECT * FROM read_csv_auto('{path}', header=true, all_varchar=true) WHERE qc_result IN ('T','F')").fetchdf()
    dfs.append(df)
manual = pd.concat(dfs, ignore_index=True)
manual['label'] = (manual['qc_result'] == 'T').astype(int)
print(f"人工标注: {len(manual)} 条 | T={manual.label.sum()} | 通过率={manual.label.mean()*100:.1f}%")

# 构建特征
X_manual = build_features(manual)
y_manual = manual['label'].values
print(f"特征数: {len(FEATURE_NAMES)}")

# 5-fold CV
model = lgb.LGBMClassifier(
    n_estimators=300, learning_rate=0.05, num_leaves=31,
    min_child_samples=5, subsample=0.8, colsample_bytree=0.8,
    random_state=RANDOM_SEED, verbose=-1,
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
cv_pred = cross_val_predict(model, X_manual, y_manual, cv=cv, method='predict_proba')[:, 1]

# CV 指标
from sklearn.metrics import roc_auc_score, precision_recall_curve
cv_auc = roc_auc_score(y_manual, cv_pred)
print(f"\n5-fold CV AUC: {cv_auc:.3f}")

# CV Precision@TopK
test_df_cv = pd.DataFrame({'proba': cv_pred, 'true': y_manual})
for k in [10, 20, 30, 50]:
    n = max(1, int(len(test_df_cv) * k / 100))
    top = test_df_cv.nlargest(n, 'proba')
    print(f"  CV Top {k:>2}%: precision={top['true'].mean()*100:.1f}%  recall={top['true'].sum()/y_manual.sum()*100:.1f}%")

# 最终用全量标注数据训练模型（用于生产打分）
model.fit(X_manual, y_manual)
print("\n✅ 生产模型已训练 (全量人工标注)")

# Feature importance
print(f"\n Feature Importance:")
imp = pd.DataFrame({'feature': FEATURE_NAMES, 'importance': model.feature_importances_})
imp = imp.sort_values('importance', ascending=False).head(12)
for _, row in imp.iterrows():
    bar = '█' * int(row['importance'] / imp['importance'].max() * 40)
    print(f"  {row['feature']:<20} {row['importance']:.4f} {bar}")

# ══════════════════════════════════════
# Step 2: 全量 PASS 打分
# ══════════════════════════════════════
print(f"\n{'='*60}")
print(f"Step 2: 全量 PASS 打分 (194K 条)")
print(f"{'='*60}")

all_pass = con.execute(f"SELECT * FROM read_csv_auto('{ALL_PASS_PATH}', header=true, all_varchar=true)").fetchdf()
con.close()
print(f"加载: {len(all_pass):,} 条")

X_all = build_features(all_pass)
all_pass['ml_score'] = model.predict_proba(X_all)[:, 1]
print(f"打分完成: score 范围 [{all_pass['ml_score'].min():.3f}, {all_pass['ml_score'].max():.3f}]")

# ══════════════════════════════════════
# Step 3: 分层切桶
# ══════════════════════════════════════
print(f"\n{'='*60}")
print(f"Step 3: 分层切桶")
print(f"{'='*60}")

all_pass = all_pass.sort_values('ml_score', ascending=False).reset_index(drop=True)
n_total = len(all_pass)

# 按分数分位
q10 = all_pass['ml_score'].quantile(0.90)
q20 = all_pass['ml_score'].quantile(0.80)
q80 = all_pass['ml_score'].quantile(0.20)

top10 = all_pass[all_pass['ml_score'] >= q10]
top20 = all_pass[(all_pass['ml_score'] >= q20) & (all_pass['ml_score'] < q10)]
mid60 = all_pass[(all_pass['ml_score'] >= q80) & (all_pass['ml_score'] < q20)]
bottom20 = all_pass[all_pass['ml_score'] < q80]

buckets = {
    'top10pct': top10,
    'top20pct': top20,
    'mid60pct': mid60,
    'bottom20pct': bottom20,
}

for name, bucket in buckets.items():
    print(f"  {name:<15}: {len(bucket):>8,} 条  avg_score={bucket['ml_score'].mean():.3f}")

# ══════════════════════════════════════
# Step 4: 抽样 + 导出
# ══════════════════════════════════════
print(f"\n{'='*60}")
print(f"Step 4: 抽样 + 导出 manual_review.csv")
print(f"{'='*60}")

np.random.seed(RANDOM_SEED)
review_rows = []

for bucket_name, n_sample in SAMPLE_SIZES.items():
    bucket = buckets[bucket_name]
    n = min(n_sample, len(bucket))
    sampled = bucket.sample(n=n, random_state=RANDOM_SEED)
    sampled = sampled.copy()
    sampled['sample_bucket'] = bucket_name
    review_rows.append(sampled)
    print(f"  {bucket_name}: 抽取 {n} 条")

review_df = pd.concat(review_rows, ignore_index=True)

# 导出字段（不含模型预测，避免人工偏见）
export_cols = [
    'sample_bucket', 'title', 'description', 'channel',
    'duration_seconds', 'video_id', 'url', 'qc_result', 'keyword'
]
export_df = review_df[export_cols].copy()
export_df['human_label'] = ''  # 待人工填写

review_path = f'{OUT_DIR}/manual_review.csv'
export_df.to_csv(review_path, index=False, encoding='utf-8-sig')
print(f"\n✅ 导出: {review_path}")
print(f"   共 {len(export_df)} 条 ({len(export_cols)} 列)")
print(f"   human_label 列留空，待人工标注 (1=PASS, 0=FAIL)")

# 也保存带 score 的版本（供后续分析用，不给人）
score_path = f'{OUT_DIR}/manual_review_with_scores.csv'
review_df.to_csv(score_path, index=False, encoding='utf-8-sig')
print(f"   (内部) 带分版本: {score_path}")

# ══════════════════════════════════════
# Step 5+6: 人工质检说明
# ══════════════════════════════════════
print(f"\n{'='*60}")
print(f"Step 5+6: 人工质检指南")
print(f"{'='*60}")
print(f"""
📋 文件: {review_path}

标注规则:
  1 = 真正双人对话 (两个真实人类在对话)
  0 = 非双人对话 (单人/多人/动画/音频/非对话内容)

标注方式:
  打开 CSV → human_label 列填写 1 或 0
  或分别存为 T.csv / F.csv

完成后, 回传标注文件, 运行 Step 7 统计
""")

# ══════════════════════════════════════
# 保存模型 + 全量分数
# ══════════════════════════════════════
print(f"{'='*60}")
print(f"模型保存")
print(f"{'='*60}")
import joblib
joblib.dump(model, f'{OUT_DIR}/lgbm_model.pkl')
print(f"  模型: {OUT_DIR}/lgbm_model.pkl")

# 全量分数导出（供后续分析）
score_df = all_pass[['video_id', 'title', 'channel', 'duration_seconds', 'qc_result', 'ml_score']].copy()
score_df.to_csv(f'{OUT_DIR}/all_pass_scores.csv', index=False, encoding='utf-8-sig')
print(f"  全量分数: {OUT_DIR}/all_pass_scores.csv ({len(score_df):,} 条)")
