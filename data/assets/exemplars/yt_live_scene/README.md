# yt_live_scene — YouTube「直播场景」样例原型

**样例源：** `/home/jqs/Downloads/直播和直播带货场景视频样品/YouTube/直播场景/`（10× mp4）

**用法：** 对候选 CSV 缩略图打 `sim_score`，按 high/mid/low 分层。  
**不是硬门禁：** YouTube `liveBroadcastContent` 只作旁路分析。

```bash
# 重建原型
02_脚本/tools/build_exemplar_bank.py \
  --video-dir "/home/jqs/Downloads/直播和直播带货场景视频样品/YouTube/直播场景" \
  -o data/assets/exemplars/yt_live_scene/

# 打分（默认 mid 也保留）
02_脚本/tools/score_exemplar_sim.py candidates.csv \
  --bank data/assets/exemplars/yt_live_scene/ \
  -o $BATCH/06_tools/
```

带货场景请另建 bank（锚 `YouTube/直播带货场景/`），勿与本库混分。
