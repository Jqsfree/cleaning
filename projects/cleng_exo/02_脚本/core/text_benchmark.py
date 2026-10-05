"""Retrospective metadata model comparison with fixed human, channel-isolated splits.
No model is promoted; test labels never select thresholds. Encoders load locally only.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from core.data_profile import file_hash
from core.stage_eval import decision_metrics
from core.topic_loop import human_gold, title_key, digest, write_json, write_csv, wilson, predict_scores


def validate_splits(splits: dict) -> None:
    keys={}
    for name,rows in splits.items():
        if not rows: raise ValueError("Empty split: "+name)
        keys[name]={
            "id":{r["video_id"] for r in rows},
            "title":{title_key(r["title"]) for r in rows},
            "channel":{r.get("channel","").strip().casefold() for r in rows if r.get("channel","").strip()},
        }
    names=list(keys)
    for i,a in enumerate(names):
        for b in names[i+1:]:
            for field in keys[a]:
                if keys[a][field]&keys[b][field]: raise ValueError(f"{a}/{b} {field} overlap")


def choose_thresholds(labels, scores, target, min_n):
    import numpy as np
    if not 0<target<1 or min_n<1: raise ValueError("Invalid threshold policy")
    labels=np.asarray(labels)
    scores=np.asarray(scores)
    if not np.isfinite(scores).all(): raise ValueError("Nonfinite scores")
    keeps=[];drops=[]
    for t in np.unique(scores):
        for bucket,mask,positive in ((keeps,scores>=t,"T"),(drops,scores<=t,"F")):
            n=int(mask.sum())
            if n>=min_n and wilson(int((labels[mask]==positive).sum()),n)[0]>=target:
                bucket.append(float(t))
    keep=min(keeps) if keeps else None
    drop=max(drops) if drops else None
    if keep is not None and drop is not None and drop>=keep: raise ValueError("Overlapping thresholds")
    return {"keep":keep,"drop":drop,"target":target,"minimum_calibration_rows":min_n,
            "calibration_support":{lab:int((labels==lab).sum()) for lab in ("T","F","U")},
            "best_possible_bound_given_class_count":{lab:wilson(int((labels==lab).sum()),int((labels==lab).sum()))[0] for lab in ("T","F")},
            "selection":"90% Wilson lower bound on calibration; exploratory selection, not batch acceptance"}


def text_field(row, fields):
    values=[]
    for name in fields:
        value=row.get(name,"")
        if name=="description":
            value=re.sub(r"https?://\S+", " ", value)[:1500]
        values.append(re.sub(r"\s+"," ",value).strip())
    return " ".join(v for v in values if v)


def compare_text(gold_dir, category, output, *, methods=("tfidf",), encoder=None,
                 fields=("title",), target=.95, min_n=20):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion
    from sklearn.metrics import roc_auc_score, average_precision_score
    if not set(fields)<= {"title","channel","description"} or "title" not in fields:
        raise ValueError("Features must include title; only title/channel/description allowed")
    if not methods or not set(methods)<= {"tfidf","embedding"}: raise ValueError("Unknown method")
    if "embedding" in methods and not encoder: raise ValueError("Local encoder required")
    paths={s:Path(gold_dir)/(s+".csv") for s in ("train","calibration","test")}
    splits={s:human_gold(p,category) for s,p in paths.items()}
    validate_splits(splits)
    for name in ("train","calibration"):
        if {r["topic_label"] for r in splits[name] if r["topic_label"]!="U"}!={"T","F"}:
            raise ValueError("Both human T and F needed in "+name)
    # No U labels are converted into training positives.
    splits["train"]=[r for r in splits["train"] if r["topic_label"] in ("T","F")]
    texts={s:[text_field(r,fields) for r in rows] for s,rows in splits.items()}
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    result={"category":category,"status":"retrospective_experiment_not_release_evidence",
            "features":list(fields),"split_hashes":{s:file_hash(p) for s,p in paths.items()},
            "split_counts":{s:len(r) for s,r in splits.items()},"models":{},
            "note":"Existing historical human splits; previous development exposure is not ruled out. No deployment promotion."}
    for method in methods:
        started=time.perf_counter()
        if method=="tfidf":
            encoder_obj=FeatureUnion([
                ("word",TfidfVectorizer(ngram_range=(1,2),min_df=1,max_features=25000,sublinear_tf=True)),
                ("char",TfidfVectorizer(analyzer="char_wb",ngram_range=(2,5),min_df=2,max_features=50000,sublinear_tf=True)),
            ])
            features={"train":encoder_obj.fit_transform(texts["train"])}
            features.update({s:encoder_obj.transform(texts[s]) for s in ("calibration","test")})
            encoder_info={"method":"word_char_tfidf_train_only"}
        else:
            import torch
            from sentence_transformers import SentenceTransformer
            torch.set_num_threads(2)
            enc=SentenceTransformer(str(encoder),local_files_only=True,device="cpu")
            e5="e5" in str(encoder).lower()
            encoded={s:[("query: " if e5 else "")+t for t in ts] for s,ts in texts.items()}
            token_stats={}
            for split, values in encoded.items():
                lengths=[len(ids) for ids in enc.tokenizer(values, truncation=False, padding=False, verbose=False)["input_ids"]]
                token_stats[split]={"rows":len(lengths),"over_limit_rows":sum(n>enc.max_seq_length for n in lengths),
                                    "max_tokens":max(lengths),"p50_tokens":float(np.quantile(lengths,.5)),
                                    "p95_tokens":float(np.quantile(lengths,.95))}
            features={s:enc.encode(ts,batch_size=64,normalize_embeddings=True,show_progress_bar=False) for s,ts in encoded.items()}
            encoder_info={"encoder":str(encoder),"max_seq_length":enc.max_seq_length,"prefix":"query: " if e5 else "", "device":"cpu"}
            encoder_info["token_lengths_before_truncation"]=token_stats
            # Store the exact resolved local encoder signature, not just a friendly name.
            source=Path(encoder)
            if source.is_dir():
                encoder_info["files"]={str(p.relative_to(source)):file_hash(p) for p in source.rglob('*')
                    if p.is_file() and (p.suffix in ('.json','.safetensors') or p.name=='pytorch_model.bin')}
        y=np.array([r["topic_label"]=="T" for r in splits["train"]])
        clf=LogisticRegression(C=1,class_weight="balanced",solver="liblinear",max_iter=2000,random_state=42)
        clf.fit(features["train"],y)
        scores={s:(predict_scores(clf,features[s]) if method=="embedding" else
                   clf.predict_proba(features[s])[:,list(clf.classes_).index(True)]) for s in ("calibration","test")}
        if not all(np.isfinite(v).all() for v in scores.values()): raise ValueError("Nonfinite evaluation scores")
        thresholds=choose_thresholds([r["topic_label"] for r in splits["calibration"]],scores["calibration"],target,min_n)
        decisions=[]
        for row,score in zip(splits["test"],scores["test"]):
            decision="review"
            if thresholds["keep"] is not None and score>=thresholds["keep"]: decision="keep"
            elif thresholds["drop"] is not None and score<=thresholds["drop"]: decision="drop"
            decisions.append({"video_id":row["video_id"],"title_hash":title_key(row["title"]),
                              "decision":decision,"score":float(score),"topic_label":row["topic_label"]})
        binary=[dict(r,decision="keep" if r["score"]>=.5 else "drop") for r in decisions]
        mask=np.array([r["topic_label"] in ("T","F") for r in splits["test"]])
        test_y=np.array([r["topic_label"]=="T" for r in splits["test"]])[mask]
        report={"thresholds":thresholds,"metrics":decision_metrics(splits["test"],decisions),
            "diagnostic_at_0_5_not_operational":decision_metrics(splits["test"],binary),
            "auc":float(roc_auc_score(test_y,scores["test"][mask])) if len(set(test_y))==2 else None,
            "ap":float(average_precision_score(test_y,scores["test"][mask])) if len(set(test_y))==2 else None,
            "seconds":time.perf_counter()-started,"encoder":encoder_info}
        result["models"][method]=report
        write_csv(output/(method+"_predictions.csv"),decisions,["video_id","title_hash","topic_label","score","decision"])
        write_json(output/"benchmark.json",result)
    return result
