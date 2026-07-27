# 排除类优化建议（自动生成）

- 生成时间: 2026-07-27 14:12:26
- 告警: proposed_without_validation, many_tags_unvalidated, validated_stalled

**重要：** 本文件只是建议。自动优化不保证准确；
准确度来自 overturn 校准 + 人工闸门。
禁止：无 validated 自训、用 proposed 当金标扩训（确认偏误）。

| level | action | target | detail | shadow |
|-------|--------|--------|--------|--------|
| info | collect_more_validation | `academic_news/text/blacklist` | n_sampled=0 < min_n=30；不可收紧自动阈值 | True |
| info | collect_more_validation | `ai_concept_trailer/text/blacklist` | n_sampled=0 < min_n=30；不可收紧自动阈值 | True |
| info | collect_more_validation | `anime_cartoon/text/blacklist` | n_sampled=0 < min_n=30；不可收紧自动阈值 | True |
| info | collect_more_validation | `audiobook_audio_only/text/blacklist` | n_sampled=0 < min_n=30；不可收紧自动阈值 | True |
| info | collect_more_validation | `award_show/text/blacklist` | n_sampled=0 < min_n=30；不可收紧自动阈值 | True |
| info | collect_more_validation | `behind_scenes/text/blacklist` | n_sampled=0 < min_n=30；不可收紧自动阈值 | True |
| info | collect_more_validation | `celebrity_gossip/text/blacklist` | n_sampled=0 < min_n=30；不可收紧自动阈值 | True |
| info | collect_more_validation | `commentary_review/text/blacklist` | n_sampled=0 < min_n=30；不可收紧自动阈值 | True |
| info | collect_more_validation | `compilation_topn/text/blacklist` | n_sampled=0 < min_n=30；不可收紧自动阈值 | True |
| info | collect_more_validation | `demo_test_video/text/blacklist` | n_sampled=0 < min_n=30；不可收紧自动阈值 | True |
| info | collect_more_validation | `documentary/text/blacklist` | n_sampled=0 < min_n=30；不可收紧自动阈值 | True |
| info | collect_more_validation | `download_tutorial/text/blacklist` | n_sampled=0 < min_n=30；不可收紧自动阈值 | True |
| info | collect_more_validation | `fashion_show/text/blacklist` | n_sampled=0 < min_n=30；不可收紧自动阈值 | True |
| info | collect_more_validation | `gaming/text/blacklist` | n_sampled=0 < min_n=30；不可收紧自动阈值 | True |
| info | collect_more_validation | `health_medical/text/blacklist` | n_sampled=0 < min_n=30；不可收紧自动阈值 | True |
| info | collect_more_validation | `interview_actor/text/blacklist` | n_sampled=0 < min_n=30；不可收紧自动阈值 | True |
| info | collect_more_validation | `kids_nursery/text/blacklist` | n_sampled=0 < min_n=30；不可收紧自动阈值 | True |
| info | collect_more_validation | `movie_clip_scene/text/blacklist` | n_sampled=0 < min_n=30；不可收紧自动阈值 | True |
| info | collect_more_validation | `movie_explainer/text/blacklist` | n_sampled=0 < min_n=30；不可收紧自动阈值 | True |
| info | collect_more_validation | `music_official/text/blacklist` | n_sampled=0 < min_n=30；不可收紧自动阈值 | True |
| info | collect_more_validation | `news_politics/text/blacklist` | n_sampled=0 < min_n=30；不可收紧自动阈值 | True |
| info | collect_more_validation | `non_english_chinese_language/text/blacklist` | n_sampled=0 < min_n=30；不可收紧自动阈值 | True |
| info | collect_more_validation | `non_english_film/text/blacklist` | n_sampled=0 < min_n=30；不可收紧自动阈值 | True |
| info | collect_more_validation | `non_film_channel/text/blacklist` | n_sampled=0 < min_n=30；不可收紧自动阈值 | True |
| info | collect_more_validation | `podcast_talk_show/text/blacklist` | n_sampled=0 < min_n=30；不可收紧自动阈值 | True |
| info | collect_more_validation | `prank_challenge/text/blacklist` | n_sampled=0 < min_n=30；不可收紧自动阈值 | True |
| info | collect_more_validation | `provisional:thumb_fail/thumb/vision_thumb` | n_sampled=0 < min_n=30；不可收紧自动阈值 | True |
| info | collect_more_validation | `radio_podcast/text/blacklist` | n_sampled=0 < min_n=30；不可收紧自动阈值 | True |
| info | collect_more_validation | `ranking_list/text/blacklist` | n_sampled=0 < min_n=30；不可收紧自动阈值 | True |
| info | collect_more_validation | `reaction_review/text/blacklist` | n_sampled=0 < min_n=30；不可收紧自动阈值 | True |
| info | collect_more_validation | `reality_docu_show/text/blacklist` | n_sampled=0 < min_n=30；不可收紧自动阈值 | True |
| info | collect_more_validation | `religious/text/blacklist` | n_sampled=0 < min_n=30；不可收紧自动阈值 | True |
| info | collect_more_validation | `shopping_haul/text/blacklist` | n_sampled=0 < min_n=30；不可收紧自动阈值 | True |
| info | collect_more_validation | `short_drama_tags/text/blacklist` | n_sampled=0 < min_n=30；不可收紧自动阈值 | True |
| info | collect_more_validation | `sports_event/text/blacklist` | n_sampled=0 < min_n=30；不可收紧自动阈值 | True |
| info | collect_more_validation | `tech_review/text/blacklist` | n_sampled=0 < min_n=30；不可收紧自动阈值 | True |
| info | collect_more_validation | `tutorial_howto/text/blacklist` | n_sampled=0 < min_n=30；不可收紧自动阈值 | True |
| info | collect_more_validation | `tv_series_episode/text/blacklist` | n_sampled=0 < min_n=30；不可收紧自动阈值 | True |
| info | collect_more_validation | `variety_comedy_show/text/blacklist` | n_sampled=0 < min_n=30；不可收紧自动阈值 | True |
| info | collect_more_validation | `vlog_challenge/text/blacklist` | n_sampled=0 < min_n=30；不可收紧自动阈值 | True |
| info | collect_more_validation | `vlog_lifestyle/text/blacklist` | n_sampled=0 < min_n=30；不可收紧自动阈值 | True |
| critical | deadlock_alert | `proposed_without_validation` | 见 metrics.deadlock_alerts；停止扩大 auto-propose，先补抽样验证 | False |
| critical | deadlock_alert | `many_tags_unvalidated` | 见 metrics.deadlock_alerts；停止扩大 auto-propose，先补抽样验证 | False |
| critical | deadlock_alert | `validated_stalled` | 见 metrics.deadlock_alerts；停止扩大 auto-propose，先补抽样验证 | False |

## 应用方式

1. 人工审阅上表
2. 手动改 `categories/_shared/reject_cascade.toml` 的 sources / 阈值
3. 或：`suggest_reject_opt.py --apply --i-understand` 仅写入 **shadow 副本**
