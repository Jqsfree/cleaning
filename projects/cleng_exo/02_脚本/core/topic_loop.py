"""Title-only topic loop. Human topic labels are separate from video-quality labels."""
from __future__ import annotations
import csv, hashlib, json, math, random, re, unicodedata
from pathlib import Path
from collections import Counter

def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

def title_key(title):
    return hashlib.sha256(" ".join(unicodedata.normalize("NFKC", title).casefold().split()).encode()).hexdigest()

def read_rows(path):
    from core.data_profile import iter_metadata
    for row in iter_metadata(path):
        if not row["video_id"].strip(): raise ValueError("Empty video_id")
        row["video_id"] = row["video_id"].strip()
        yield row

def write_csv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")

def load_policy(path):
    p = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("category", "definition", "version", "related_patterns", "unrelated_patterns"):
        if key not in p:
            raise ValueError("Missing policy field: "+key)
    for group in ("related_patterns", "unrelated_patterns"):
        for rule in p[group]:
            re.compile(rule["pattern"], re.I)
    return p

def classify(title, policy):
    if not title.strip():
        return "review", "missing_title"
    pos = [r["id"] for r in policy["related_patterns"] if re.search(r["pattern"], title, re.I)]
    neg = [r["id"] for r in policy["unrelated_patterns"] if re.search(r["pattern"], title, re.I)]
    if pos and neg:
        return "review", "conflicting_evidence:"+",".join(pos+neg)
    if neg:
        return "drop", ",".join(neg)
    if pos:
        return "keep", ",".join(pos)
    return "review", "insufficient_title_evidence"

def human_gold(path, category):
    return validate_human_rows(read_rows(path),category)

def validate_human_rows(rows, category):
    result, seen, titles = [], {}, {}
    for row in rows:
        if row.get("topic_label") not in ("T", "F", "U"):
            raise ValueError("Explicit topic_label T/F/U required; old qc_result is NOT topic gold")
        if row.get("label_source") != "human" or not row.get("reviewer", "").strip():
            raise ValueError("Human label_source and reviewer required")
        if row.get("topic_category") != category:
            raise ValueError("Human topic category mismatch")
        key = title_key(row["title"])
        if row["video_id"] in seen and seen[row["video_id"]] != row["topic_label"]:
            raise ValueError("Conflicting video labels")
        if key in titles and titles[key] != row["topic_label"]:
            raise ValueError("Identical titles have conflicting topic labels; adjudication required")
        if row["video_id"] not in seen:
            result.append(row)
        seen[row["video_id"]] = row["topic_label"]
        titles[key] = row["topic_label"]
    return result

def wilson(success, n, z=1.6448536269514722):
    if not n:
        return [None, None]
    p = success/n
    center = (p+z*z/(2*n))/(1+z*z/n)
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [max(0,center-half), min(1,center+half)]

def sample_size(margin=.05, confidence=90):
    z = {90:1.6448536269514722,95:1.959963984540054}[confidence]
    return math.ceil(z*z*.25/margin**2)

def predict_scores(classifier, features):
    import numpy as np
    from scipy.special import expit
    x=np.asarray(features,dtype=np.float64)
    if not np.isfinite(x).all() or not np.isfinite(classifier.coef_).all():
        raise ValueError("Non-finite embeddings or classifier")
    scores=expit(np.einsum("ij,j->i",x,classifier.coef_[0],optimize=False)+classifier.intercept_[0])
    if not np.isfinite(scores).all(): raise ValueError("Non-finite scores")
    return scores

def decide_title(title, policy, model=None, probability=None):
    if model is None or not title.strip(): return classify(title,policy)
    if probability >= model["keep_threshold"]: return "keep", "human_topic_model"
    if probability <= model["drop_threshold"]: return "drop", "human_topic_model"
    return "review", "model_uncertain"

def run_round(input_path, policy_path, out, sample_n=271, seed=42, model_path=None, verifier_report=None,
              *, target=None, purpose="candidate", parent_round=None, reference_gold=None, max_t_loss=None, audit_sizes=None):
    from core.data_profile import profile_file, write_profile, file_hash
    from core.metadata_acceptance import AUDIT_FIELDS, metadata_hash, candidate_id, frozen_manifest
    if sample_n < 1: raise ValueError("sample_n must be positive")
    if max_t_loss is not None and not 0<=max_t_loss<1: raise ValueError("Invalid T loss limit")
    audit_sizes = audit_sizes or {p:sample_n for p in ("keep","drop","review")}
    if set(audit_sizes)!={"keep","drop","review"} or any(type(n) is not int or n<1 for n in audit_sizes.values()):
        raise ValueError("Positive fixed audit sizes required for all three pools")
    if target is not None and not 0 < target < 1: raise ValueError("Invalid acceptance target")
    if purpose not in ("candidate", "regression"): raise ValueError("Invalid purpose")
    if reference_gold and verifier_report: raise ValueError("Reference replay requires rules or local model, not an external verifier")
    policy = load_policy(policy_path)
    parent_candidate=None
    parent_development=set()
    if parent_round:
        parent=frozen_manifest(parent_round)
        from core.metadata_acceptance import verified_evaluation, load_audit_gold
        report=verified_evaluation(parent_round,parent)
        if report["status"]!="failed_acceptance" or parent["category"]!=policy["category"]:
            raise ValueError("Specialist requires failed human acceptance in the same category")
        if parent["acceptance_plan"]["target"]!=target or parent["acceptance_plan"].get("max_t_loss")!=max_t_loss:
            raise ValueError("Specialist must preserve the failed batch acceptance targets")
        labels=Path(parent_round)/report["labels_snapshot"]
        parent_candidate=parent["candidate_id"]
        parent_development={title_key(r["title"]) for r in load_audit_gold(labels,parent)}
        if file_hash(input_path)!=parent["input_sha256"]:
            raise ValueError("Specialist must use the same batch input snapshot")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    profile=profile_file(input_path,policy["category"])
    write_profile(profile,out/"intake")
    if profile["status"]!="execution_checks_passed":
        raise ValueError("Input contract blocked; inspect intake/profile.json")
    pools = ("keep","drop","review")
    handles, writers = {}, {}
    reservoirs = {x:[] for x in pools}
    counts = Counter()
    rngs = {x:random.Random(str(seed)+x) for x in pools}
    verifier = None
    if verifier_report:
        verifier=json.loads(Path(verifier_report).read_text())
        actual=hashlib.sha256(Path(input_path).read_bytes()).hexdigest()
        if actual!=verifier.get("verified_sha256") or verifier["policy_hash"]!=digest(policy):
            raise ValueError("Verifier input/policy fingerprint mismatch")
        if model_path: raise ValueError("Use model scoring or verified decisions, not both")
    model = None
    model_sha = file_hash(model_path) if model_path else None
    if model_path:
        import joblib
        model = joblib.load(model_path)
        if model["category"] != policy["category"] or model["policy_hash"] != digest(policy):
            raise ValueError("Model policy mismatch: retrain/recalibrate")
    enc = None
    if model:
        from sentence_transformers import SentenceTransformer
        enc = SentenceTransformer(model["encoder"], local_files_only=True)
    source_hash = hashlib.sha256()
    seen = set()
    duplicate = 0
    def consume(batch):
        probabilities = [None]*len(batch)
        if model:
            x = enc.encode([r["title"] for r in batch], normalize_embeddings=True, show_progress_bar=False)
            probabilities = predict_scores(model["classifier"],x)
        for row, probability in zip(batch,probabilities):
            decision, reason = decide_title(row["title"],policy,model,probability)
            if verifier:
                decision=row.get("topic_decision")
                if decision not in pools: raise ValueError("Invalid verified decision")
                reason=row.get("topic_reason","")
            record = dict(row, topic_decision=decision, topic_reason=reason,
                topic_score="" if probability is None else float(probability),
                topic_category=policy["category"], topic_title_hash=title_key(row["title"]))
            writers[decision].writerow(record)
            counts[decision] += 1
            bucket = reservoirs[decision]
            if len(bucket)<audit_sizes[decision]:
                bucket.append(record)
            else:
                j = rngs[decision].randrange(counts[decision])
                if j<audit_sizes[decision]:
                    bucket[j]=record
    try:
        batch=[]
        for row in read_rows(input_path):
            source_hash.update(json.dumps(row,sort_keys=True,ensure_ascii=False).encode())
            vid=row["video_id"].strip()
            if vid in seen:
                duplicate+=1
                continue
            seen.add(vid)
            if not writers:
                reserved={"topic_decision","topic_reason","topic_score","topic_category","topic_title_hash"}
                fields=[k for k in row if k not in reserved]+sorted(reserved)
                for name in pools:
                    handles[name]=(out/(name+".csv")).open("w",encoding="utf-8-sig",newline="")
                    writers[name]=csv.DictWriter(handles[name],fieldnames=fields,extrasaction="ignore")
                    writers[name].writeheader()
            batch.append(row)
            if len(batch)>=512:
                consume(batch); batch=[]
        if batch:
            consume(batch)
        if not writers:
            raise ValueError("Empty input")
    finally:
        for h in handles.values(): h.close()
    audit_fields=AUDIT_FIELDS
    expected={};audit_metadata={};combined=[]
    human_fields={"topic_label","label_source","reviewer","unrelated_topic","review_notes"}
    for name in pools:
        rows=[{k:("" if k in human_fields else r.get(k,"")) for k in audit_fields} for r in reservoirs[name]]
        write_csv(out/("audit_"+name+".csv"),rows,audit_fields)
        expected[name]={r["video_id"]:r["topic_title_hash"] for r in rows}
        audit_metadata.update({r["video_id"]:metadata_hash(r) for r in rows})
        combined.extend(rows)
    random.Random(seed).shuffle(combined)
    write_csv(out/"audit.csv",combined,audit_fields)
    if file_hash(input_path)!=profile["input_sha256"]: raise ValueError("Input changed during run")
    if model_path and file_hash(model_path)!=model_sha: raise ValueError("Model changed during run")
    if sum(counts.values())!=len(seen): raise ValueError("Decision conservation failed")
    manifest={"category":policy["category"],"policy":policy,"policy_hash":digest(policy),
        "input":str(Path(input_path).resolve()),"source_hash":source_hash.hexdigest(),
        "counts":dict(counts),"unique_rows":len(seen),"duplicates_skipped":duplicate,
        "sample_size_requested":sample_n,"seed":seed,"audit_expected":expected,
        "model_path":str(model_path) if model_path else None,
        "model_hash":model_sha,
        "development_titles":sorted(parent_development | set(policy.get("development_titles",[])) | set(model["development_titles"] if model else []) | set(verifier["development_titles"] if verifier else [])),
        "verifier_hash":digest({k:verifier[k] for k in ("policy_hash","gold_hash","model","prompt_version")}) if verifier else None,
        "status":"awaiting_independent_human_topic_audit",
        "note":"keep is a machine candidate pool, not verified title purity"}
    reference_report=None
    if reference_gold:
        from core.stage_eval import evaluate_trace
        gold=human_gold(reference_gold,policy["category"])
        if not gold: raise ValueError("Empty human reference")
        probabilities=predict_scores(model["classifier"],enc.encode([r["title"] for r in gold],
            normalize_embeddings=True,show_progress_bar=False)) if model else [None]*len(gold)
        reference_decisions=[]
        for row,score in zip(gold,probabilities):
            dec,reason=decide_title(row["title"],policy,model,score)
            reference_decisions.append(dict(video_id=row["video_id"],title_hash=title_key(row["title"]),decision=dec,reason=reason))
        manifest["development_titles"]=sorted(set(manifest["development_titles"]) | {title_key(r["title"]) for r in gold})
        reference_report=evaluate_trace(gold,[("topic_classifier",reference_decisions)])
        reference_report["reference_sha256"]=file_hash(reference_gold)
        write_json(out/"reference_eval.json",reference_report)
        write_csv(out/"reference_decisions.csv",reference_decisions,["video_id","title_hash","decision","reason"])
    manifest.update(schema_version=2,purpose=purpose,parent_candidate=parent_candidate,
        input_sha256=profile["input_sha256"],audit_metadata=audit_metadata,
        acceptance_plan={"target":target,"confidence":90,"method":"wilson_interval_or_census","U":"non_pass",
                         "max_t_loss":max_t_loss,"audit_sizes":audit_sizes},
        pool_hashes={name:file_hash(out/(name+".csv")) for name in pools})
    manifest["candidate_id"]=candidate_id(manifest)
    write_json(out/"stage_eval.json",{"status":"execution_checks_passed","unique_input":len(seen),
        "pool_counts":dict(counts),"duplicates_skipped":duplicate,"semantic_evaluation":"reference_eval.json" if reference_report else "requires_human_reference_or_audit"})
    write_json(out/"manifest.json",manifest)
    return {k:v for k,v in manifest.items() if k not in ("audit_expected","audit_metadata","development_titles","policy")}

def evaluate(round_dir, labels_path, target=None):
    from core.metadata_acceptance import evaluate_round
    return evaluate_round(round_dir, labels_path, target)


def train_topic(train_path, calibration_path, policy_path, output, encoder):
    import joblib
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    p=load_policy(policy_path)
    train=human_gold(train_path,p["category"])
    calibration=human_gold(calibration_path,p["category"])
    from core.text_benchmark import validate_splits
    validate_splits({"train":train,"calibration":calibration})
    if {title_key(r["title"]) for r in train}&{title_key(r["title"]) for r in calibration}:
        raise ValueError("Train/calibration title overlap")
    train=[r for r in train if r["topic_label"]!="U"]
    if any(sum(r["topic_label"]==lab for r in group)<20 for group in (train,calibration) for lab in ("T","F")):
        raise ValueError("Need >=20 explicit human T and F in each development split")
    enc=SentenceTransformer(encoder,local_files_only=True)
    x=enc.encode([r["title"] for r in train],normalize_embeddings=True,show_progress_bar=False)
    if not np.isfinite(x).all(): raise ValueError("Invalid embeddings")
    clf=LogisticRegression(C=1,class_weight="balanced",solver="liblinear",max_iter=2000,random_state=42).fit(
        x,np.array([r["topic_label"]=="T" for r in train],dtype=int))
    scores=predict_scores(clf,enc.encode([r["title"] for r in calibration],normalize_embeddings=True,show_progress_bar=False))
    y=np.array([r["topic_label"]=="T" for r in calibration])
    from core.text_benchmark import choose_thresholds
    selected=choose_thresholds([r["topic_label"] for r in calibration],scores,.95,20)
    keep=selected["keep"] if selected["keep"] is not None else 1.01
    drop=selected["drop"] if selected["drop"] is not None else -.01
    bundle={"category":p["category"],"policy_hash":digest(p),"encoder":encoder,"classifier":clf,
        "keep_threshold":keep,"drop_threshold":drop,"threshold_selection":selected,
        "development_titles":sorted({title_key(r["title"]) for r in train+calibration}),
        "status":"candidate_requires_fresh_human_audit","feature_fields":["title"]}
    output=Path(output)
    if output.exists(): raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(bundle,output)
    write_json(output.with_suffix(".json"),{k:v for k,v in bundle.items() if k not in ("classifier","development_titles")})
    return {k:v for k,v in bundle.items() if k not in ("classifier","development_titles")}

def propose(labels, policy_path, out, validation_path=None):
    from core.text_boundary import propose_rules
    p=load_policy(policy_path)
    gold=human_gold(labels,p["category"])
    rows=[dict(r,channel="",qc_text_result=r["topic_label"]) for r in gold]
    proposals=propose_rules(rows,None,min_f=3)
    payload={"category":p["category"],"parent_policy_hash":digest(p),
        "status":"proposals_only_not_active","rules":[r.as_dict() for r in proposals],
        "note":"Review topic meaning and regression-test all human topic T/U before activating. Fresh audit required after every policy change."}
    from core.data_profile import file_hash
    payload["development_sha256"]=file_hash(labels)
    payload["development_titles"]=sorted({title_key(r["title"]) for r in gold})
    if validation_path:
        from core.rule_validation import validate_proposals
        validation=human_gold(validation_path,p["category"])
        payload["validation"]=validate_proposals(gold,validation,payload["rules"],p)
        payload["validation_sha256"]=file_hash(validation_path)
        payload["development_titles"]=sorted(set(payload["development_titles"])|{title_key(r["title"]) for r in validation})
    else:
        payload["validation_status"]="missing_independent_validation"
    validation_status={r["name"]:r["status"] for r in payload.get("validation",{}).get("rules",[])}
    for rule in payload["rules"]:
        rule["development_addable"]=rule.pop("addable",False)
        rule["validation_status"]=validation_status.get(rule["name"],"missing_independent_validation")
        rule["eligible_for_revision"]=rule["validation_status"]=="candidate_for_review_not_accepted"
    if Path(out).exists(): raise FileExistsError(out)
    write_json(out,payload)
    return payload

def import_metadata_gold(source_dir, category, out, seed=42):
    """User-confirmed metadata human QC; never import LLM files or playback-only F."""
    out=Path(out); out.mkdir(parents=True,exist_ok=False)
    by_id={}; excluded=[]; conflicts=[]
    for path in sorted(Path(source_dir).glob("*_qc_result.csv")):
        with path.open(encoding="utf-8-sig",newline="") as handle:
            for row in csv.DictReader(handle):
                lab=(row.get("qc_result") or "").strip().upper()
                if None in row or not row.get("video_id") or not row.get("title") or lab not in ("T","F","U"):
                    excluded.append({"source":str(path),"video_id":row.get("video_id"),"label":lab})
                    continue
                r=dict(row,topic_label=lab,label_source="human",reviewer="existing_metadata_human_qc",
                       topic_category=category,label_file=str(path.resolve()))
                by_id.setdefault(row["video_id"],[]).append(r)
    candidates=[]
    for vid, rows in by_id.items():
        if len({r["topic_label"] for r in rows})>1:
            conflicts.extend(rows)
        else:
            candidates.append(rows[0])
    by_title={}
    for row in candidates: by_title.setdefault(title_key(row["title"]),[]).append(row)
    clean=[]
    for rows in by_title.values():
        if len({r["topic_label"] for r in rows})>1: conflicts.extend(rows)
        else: clean.append(rows[0])
    # Channel groups stay together; identical titles have already been deduplicated.
    groups={}
    for r in clean:
        group=r.get("channel","").strip().casefold() or title_key(r["title"])
        groups.setdefault(group,[]).append(r)
    keys=sorted(groups); random.Random(seed).shuffle(keys)
    splits={"train":[],"calibration":[],"test":[]}
    n=len(keys)
    for i,k in enumerate(keys):
        name="train" if i<int(.6*n) else "calibration" if i<int(.8*n) else "test"
        splits[name].extend(groups[k])
    fields=["video_id","title","channel","topic_category","topic_label","label_source","reviewer","label_file","qc_result"]
    for name, rows in splits.items(): write_csv(out/(name+".csv"),rows,fields)
    write_csv(out/"conflicts.csv",conflicts,fields)
    report={"category":category,"source_dir":str(source_dir),"seed":seed,
        "scope":"existing metadata human T/F as explicitly confirmed by user",
        "split":"channel-grouped 60/20/20; exact-title dedup; no model/LLM labels",
        "counts":{k:dict(Counter(r["topic_label"] for r in v)) for k,v in splits.items()},
        "excluded":excluded,"conflict_rows":len(conflicts),"unique_titles":len(clean)}
    write_json(out/"manifest.json",report)
    return report

def benchmark(model_path, test_path, output):
    import joblib
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics import roc_auc_score
    m=joblib.load(model_path)
    rows=human_gold(test_path,m["category"])
    if set(m["development_titles"]) & {title_key(r["title"]) for r in rows}:
        raise ValueError("Test overlaps development data")
    enc=SentenceTransformer(m["encoder"],local_files_only=True)
    x=enc.encode([r["title"] for r in rows],normalize_embeddings=True,show_progress_bar=False)
    scores=predict_scores(m["classifier"],x)
    predictions=[]
    for r,s in zip(rows,scores):
        decision="keep" if s>=m["keep_threshold"] else "drop" if s<=m["drop_threshold"] else "review"
        predictions.append(dict(r,topic_score=float(s),topic_decision=decision))
    metrics={}
    for pool in ("keep","drop","review"):
        group=[r for r in predictions if r["topic_decision"]==pool]
        counts=Counter(r["topic_label"] for r in group)
        metrics[pool]={"n":len(group),"labels":dict(counts),"human_T_ci90":wilson(counts["T"],len(group))}
    mask=[i for i,r in enumerate(rows) if r["topic_label"]!="U"]
    y=[rows[i]["topic_label"]=="T" for i in mask]
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    report={"category":m["category"],"test_rows":len(rows),"metrics":metrics,
        "auc":float(roc_auc_score(y,scores[mask])) if len(set(y))==2 else None,
        "status":"offline_holdout_only_not_current_pool_acceptance",
        "keep_threshold":m["keep_threshold"],"drop_threshold":m["drop_threshold"]}
    write_json(output/"benchmark.json",report)
    write_csv(output/"predictions.csv",predictions,
        ["video_id","title","topic_label","topic_score","topic_decision","label_file"])
    return report

def revise_policy(policy_path, proposals_path, labels_path, selected, output, validation_path=None):
    from core.text_boundary import score_pattern
    p=load_policy(policy_path)
    proposals=json.loads(Path(proposals_path).read_text())
    if proposals["parent_policy_hash"]!=digest(p): raise ValueError("Stale proposals")
    gold=human_gold(labels_path,p["category"])
    rows=[dict(r,channel="",qc_text_result=r["topic_label"]) for r in gold]
    candidates={r["name"]:r for r in proposals["rules"]}
    if not selected or len(selected)!=len(set(selected)): raise ValueError("Select unique proposal names")
    for name in selected:
        if name not in candidates: raise ValueError("Unknown proposal "+name)
        rule=candidates[name]
        s=score_pattern(rows,name,rule["pattern"],min_f=3)
        if s.n_t or s.n_u or s.n_f<3: raise ValueError("Rule fails human regression")
    if validation_path is None: raise ValueError("Independent validation gold required for rule revision")
    from core.data_profile import file_hash
    from core.rule_validation import validate_proposals
    if proposals.get("development_sha256")!=file_hash(labels_path): raise ValueError("Proposal development evidence mismatch; regenerate proposals")
    if proposals.get("validation_sha256")!=file_hash(validation_path): raise ValueError("Proposal validation evidence mismatch; regenerate proposals")
    validation=human_gold(validation_path,p["category"])
    checks=validate_proposals(gold,validation,[candidates[n] for n in selected],p)
    if any(r["status"]!="candidate_for_review_not_accepted" for r in checks["rules"]):
        raise ValueError("Rule fails independent human regression/support")
    existing={r["id"] for group in ("related_patterns","unrelated_patterns") for r in p[group]}
    for name in selected:
        if name in existing: raise ValueError("Duplicate policy rule ID: "+name)
        p["unrelated_patterns"].append({"id":name,"pattern":candidates[name]["pattern"]})
    p["development_titles"]=sorted(set(p.get("development_titles",[]))|set(proposals.get("development_titles",[]))|{title_key(r["title"]) for r in validation})
    p["rule_validation"]=dict(development_sha256=file_hash(labels_path),validation_sha256=file_hash(validation_path),**checks)
    p["version"]=p["version"]+"-revision"
    p["development_titles"]=sorted(set(p.get("development_titles",[]))|{title_key(r["title"]) for r in gold})
    if Path(output).exists(): raise FileExistsError(output)
    write_json(output,p)
    return {"status":"candidate_not_promoted","policy_hash":digest(p),"rules_added":len(selected)}

def close_loop(round_dirs, output):
    if len(round_dirs)<2: raise ValueError("Need two fresh human-audited rounds")
    seen=set(); signatures=set(); reports=[]
    for path in round_dirs:
        path=Path(path)
        from core.metadata_acceptance import frozen_manifest, assess, load_audit_gold
        from core.data_profile import file_hash
        m=frozen_manifest(path)
        ev=json.loads((path/"human_topic_evaluation.json").read_text())
        if ev["candidate_id"]!=m["candidate_id"]: raise ValueError("Stale acceptance")
        labels=path/ev["labels_snapshot"]
        if file_hash(labels)!=ev["labels_sha256"] or assess(m,load_audit_gold(labels,m))["status"]!="accepted":
            raise ValueError("Invalid human acceptance evidence")
        if ev["status"]!="accepted": raise ValueError("Round has not passed human acceptance")
        if ev["policy_hash"]!=m["policy_hash"]: raise ValueError("Evaluation policy mismatch")
        signatures.add((m["category"],m["policy_hash"],m.get("model_hash"),m.get("verifier_hash"),m["source_hash"],digest(m["acceptance_plan"])))
        keys=set(m["audit_expected"]["keep"].values())
        if seen&keys: raise ValueError("Audit title overlap across rounds")
        seen|=keys;reports.append(str(path.resolve()))
    if len(signatures)!=1: raise ValueError("Rounds must share frozen policy, model, verifier and target")
    if Path(output).exists(): raise FileExistsError(output)
    result={"status":"human_verified_text_loop","rounds":reports,
        "note":"Sample-based acceptance only. New policy/model or source drift requires a fresh audit."}
    write_json(output,result)
    return result
