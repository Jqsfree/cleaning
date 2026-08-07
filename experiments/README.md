# experiments — 实验脚本

从主管道归档的实验性代码，不再作为日常入口：

- `mvp_lightgbm.py` / `pass_filter_ml.py` — tow-person LightGBM 过滤实验
- `welding_meta_bench.py` — 电焊 vision QC 元信息压测（原根目录 `meta.py`）
- `live_sell_text_classifier.py` — live_sell yb01 正负样本训 Dual TF-IDF；产出 `models/live_sell_text_clf_yb01.pkl` + 校准 JSON（抽样置信度 90% ≠ 模型阈值）
- `film_tv_text_classifier*.py` / `film_tv_thumb_train.py` / `film_tv_multimodal.py` — film_tv 小模型实验
