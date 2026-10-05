#!/usr/bin/env python3
"""Title topic loop: run -> human audit -> evaluate -> propose/train -> new run.
Usage: .venv/bin/python3 02_脚本/tools/batch_ops/topic_loop.py --help
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from core.topic_loop import run_round, evaluate, train_topic, propose, import_metadata_gold, benchmark, revise_policy, close_loop

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest="action",required=True)
    run=sub.add_parser("run")
    run.add_argument("--input",type=Path,required=True)
    run.add_argument("--policy",type=Path,required=True)
    run.add_argument("--out",type=Path,required=True)
    run.add_argument("--model",type=Path)
    run.add_argument("--verifier-report",type=Path)
    run.add_argument("--sample-size",type=int,default=271)
    run.add_argument("--seed",type=int,default=42)
    run.add_argument("--reference-gold",type=Path,help="Automatically replay the same classifier on historical human gold")
    run.add_argument("--max-t-loss",type=float)
    run.add_argument("--audit-sizes",type=int,nargs=3,metavar=("KEEP","DROP","REVIEW"))
    run.add_argument("--target",type=float,help="Freeze human acceptance target before sampling; omitted means no release")
    run.add_argument("--purpose",choices=("candidate","regression"),default="candidate")
    run.add_argument("--parent-round",type=Path,help="Specialist: requires failed human acceptance for same batch")
    ev=sub.add_parser("evaluate")
    ev.add_argument("--round",type=Path,required=True)
    ev.add_argument("--labels",type=Path,required=True)
    ev.add_argument("--target",type=float,help="Optional assertion; must match frozen target")
    tr=sub.add_parser("train")
    tr.add_argument("--train",type=Path,required=True)
    tr.add_argument("--calibration",type=Path,required=True)
    tr.add_argument("--policy",type=Path,required=True)
    tr.add_argument("--output",type=Path,required=True)
    tr.add_argument("--encoder",default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    pr=sub.add_parser("propose")
    pr.add_argument("--labels",type=Path,required=True)
    pr.add_argument("--policy",type=Path,required=True)
    pr.add_argument("--output",type=Path,required=True)
    pr.add_argument("--validation-gold",type=Path,help="Separate-channel regression evidence for proposals")
    imp=sub.add_parser("import-gold")
    imp.add_argument("--source-dir",type=Path,required=True)
    imp.add_argument("--category",required=True)
    imp.add_argument("--out",type=Path,required=True)
    imp.add_argument("--seed",type=int,default=42)
    bm=sub.add_parser("benchmark")
    bm.add_argument("--model",type=Path,required=True)
    bm.add_argument("--test",type=Path,required=True)
    bm.add_argument("--out",type=Path,required=True)
    rev=sub.add_parser("revise")
    rev.add_argument("--policy",type=Path,required=True)
    rev.add_argument("--proposals",type=Path,required=True)
    rev.add_argument("--labels",type=Path,required=True)
    rev.add_argument("--select",nargs="+",required=True)
    rev.add_argument("--output",type=Path,required=True)
    rev.add_argument("--validation-gold",type=Path,required=True)
    close=sub.add_parser("close")
    close.add_argument("--rounds",nargs="+",type=Path,required=True)
    close.add_argument("--output",type=Path,required=True)
    verify=sub.add_parser("verify")
    verify.add_argument("--input",type=Path,required=True)
    verify.add_argument("--gold",type=Path,required=True)
    verify.add_argument("--policy",type=Path,required=True)
    verify.add_argument("--out",type=Path,required=True)
    verify.add_argument("--model",default="qwen-plus")
    verify.add_argument("--limit",type=int,default=0)
    profile=sub.add_parser("profile",help="Strict metadata intake profile")
    profile.add_argument("--input",type=Path,required=True)
    profile.add_argument("--category",required=True)
    profile.add_argument("--out",type=Path,required=True)
    profile.add_argument("--reference",type=Path)
    replay=sub.add_parser("replay",help="Replay human reference through category rules; no release")
    replay.add_argument("--gold",type=Path,required=True)
    replay.add_argument("--policy",type=Path,required=True)
    replay.add_argument("--out",type=Path,required=True)
    enrich=sub.add_parser("enrich-gold",help="Recover metadata from recorded human sources; preserve labels and splits")
    enrich.add_argument("--gold-dir",type=Path,required=True)
    enrich.add_argument("--category",required=True)
    enrich.add_argument("--out",type=Path,required=True)
    diagnose=sub.add_parser("diagnose",help="Rule hits, false keeps/drops, channel/source slices and veto impact")
    diagnose.add_argument("--gold",type=Path,required=True)
    diagnose.add_argument("--policy",type=Path,required=True)
    diagnose.add_argument("--probes",type=Path,help="Diagnostic metadata patterns; never activated as rules")
    diagnose.add_argument("--out",type=Path,required=True)
    non_live_audit=sub.add_parser("non-live-audit",help="Replay shared non-live title candidates over human annotation sources")
    non_live_audit.add_argument("--root",type=Path,action="append",default=[])
    non_live_audit.add_argument("--file",type=Path,action="append",default=[])
    non_live_audit.add_argument("--policy",type=Path)
    non_live_audit.add_argument("--out",type=Path,required=True)
    non_live_scan=sub.add_parser("non-live-scan",help="Shadow-scan a metadata pool; no rows are dropped")
    non_live_scan.add_argument("--input",type=Path,required=True)
    non_live_scan.add_argument("--policy",type=Path)
    non_live_scan.add_argument("--out",type=Path,required=True)
    topic_distribution=sub.add_parser("topic-distribution",help="Discover title topic distribution in a CSV/Parquet; no cleaning")
    topic_distribution.add_argument("--input",type=Path,required=True)
    topic_distribution.add_argument("--out",type=Path,required=True)
    topic_distribution.add_argument("--topics",type=int,default=20)
    topic_distribution.add_argument("--sample-size",type=int,default=10000)
    topic_distribution.add_argument("--batch-size",type=int,default=2048)
    topic_distribution.add_argument("--max-features",type=int,default=25000)
    topic_distribution.add_argument("--seed",type=int,default=42)
    topic_distribution.add_argument("--encoder",help="Existing local sentence-transformer directory")
    compare=sub.add_parser("compare",help="Fixed-split TF-IDF/local embedding experiment")
    compare.add_argument("--gold-dir",type=Path,required=True)
    compare.add_argument("--category",required=True)
    compare.add_argument("--out",type=Path,required=True)
    compare.add_argument("--methods",nargs="+",choices=("tfidf","embedding"),default=["tfidf"])
    compare.add_argument("--encoder",help="Local encoder checkpoint; never downloads")
    compare.add_argument("--fields",nargs="+",choices=("title","channel","description"),default=["title"])
    compare.add_argument("--target",type=float,required=True,help="Experimental calibration constraint, not a batch SLA")
    compare.add_argument("--min-calibration-n",type=int,default=20)
    release=sub.add_parser("release",help="Export only a frozen, human-accepted candidate")
    release.add_argument("--round",type=Path,required=True)
    release.add_argument("--out",type=Path,required=True)
    nxt=sub.add_parser("next",help="Show required action; never starts a new unapproved rule")
    nxt.add_argument("--round",type=Path,required=True)
    start=sub.add_parser("start",help="Create a managed batch and its first frozen candidate")
    start.add_argument("--input",type=Path,required=True)
    start.add_argument("--policy",type=Path,required=True)
    start.add_argument("--batch",required=True)
    start.add_argument("--source",choices=("human","machine"),default="machine")
    start.add_argument("--runs-root",type=Path,default=Path(__file__).resolve().parents[2]/"data/runs")
    start.add_argument("--target",type=float,required=True)
    start.add_argument("--max-t-loss",type=float)
    start.add_argument("--sample-size",type=int,default=271)
    start.add_argument("--audit-sizes",type=int,nargs=3,metavar=("KEEP","DROP","REVIEW"))
    start.add_argument("--seed",type=int,default=42)
    start.add_argument("--model",type=Path)
    start.add_argument("--reference-gold",type=Path)
    start.add_argument("--purpose",choices=("candidate","regression"),default="candidate")
    status=sub.add_parser("status",help="Verify current batch evidence and required next action")
    status.add_argument("--batch-root",type=Path,required=True)
    status.add_argument("--quick",action="store_true",help="Do not hash large files; never reports release eligibility")
    for name in ("retry","publish"):
        cmd=sub.add_parser(name)
        cmd.add_argument("--batch-root",type=Path,required=True)
    specialist=sub.add_parser("specialist",help="Requires failed acceptance; preserves batch targets")
    specialist.add_argument("--batch-root",type=Path,required=True)
    specialist.add_argument("--policy",type=Path,required=True)
    specialist.add_argument("--model",type=Path)
    specialist.add_argument("--reference-gold",type=Path)
    cat=sub.add_parser("catalog",help="Rebuild batch index without copying or moving data")
    cat.add_argument("--runs-root",type=Path,default=Path(__file__).resolve().parents[2]/"data/runs")
    cat.add_argument("--category")
    cat.add_argument("--verify",action="store_true")
    cat.add_argument("--out",type=Path,help="Optional JSON index snapshot")
    check=sub.add_parser("doctor",help="Validate category TOML and metadata policies")
    check.add_argument("--project-root",type=Path,default=Path(__file__).resolve().parents[2])
    args=ap.parse_args(argv)
    try:
        if args.action=="doctor":
            from core.metadata_workflow import doctor
            result=doctor(args.project_root)
        elif args.action=="start":
            from core.metadata_workflow import start_batch
            result=start_batch(args.input,args.policy,args.runs_root,args.batch,source=args.source,target=args.target,
                max_t_loss=args.max_t_loss,sample_size=args.sample_size,audit_sizes=dict(zip(("keep","drop","review"),args.audit_sizes)) if args.audit_sizes else None,
                seed=args.seed,model=args.model,reference_gold=args.reference_gold,purpose=args.purpose)
        elif args.action in ("status","retry","specialist","publish","catalog"):
            from core.metadata_workflow import batch_status,retry_batch,specialist_batch,publish_batch,catalog
            if args.action=="status": result=batch_status(args.batch_root,verify=not args.quick)
            elif args.action=="retry": result=retry_batch(args.batch_root)
            elif args.action=="specialist": result=specialist_batch(args.batch_root,args.policy,model=args.model,reference_gold=args.reference_gold)
            elif args.action=="publish": result=publish_batch(args.batch_root)
            else:
                result=catalog(args.runs_root,category=args.category,verify=args.verify)
                if args.out:
                    from core.runtime_files import atomic_json
                    atomic_json(args.out,result)
        elif args.action=="run":
            if args.sample_size<1: raise ValueError("sample-size must be positive")
            result=run_round(args.input,args.policy,args.out,args.sample_size,args.seed,args.model,args.verifier_report,
                target=args.target,purpose=args.purpose,parent_round=args.parent_round,reference_gold=args.reference_gold,
                max_t_loss=args.max_t_loss,audit_sizes=dict(zip(("keep","drop","review"),args.audit_sizes)) if args.audit_sizes else None)
        elif args.action=="profile":
            from core.data_profile import profile_file,write_profile
            result=profile_file(args.input,args.category,reference=args.reference)
            write_profile(result,args.out)
        elif args.action=="replay":
            from core.stage_eval import replay_rules
            result=replay_rules(args.gold,args.policy,args.out)
        elif args.action=="enrich-gold":
            from core.gold_diagnostics import enrich_gold
            result=enrich_gold(args.gold_dir,args.category,args.out)
        elif args.action=="diagnose":
            from core.gold_diagnostics import diagnose_rules
            result=diagnose_rules(args.gold,args.policy,args.out,args.probes)
        elif args.action=="non-live-audit":
            from core.non_live_text import DEFAULT_POLICY,audit_annotations
            if not args.root and not args.file: raise ValueError("non-live-audit requires --root or --file")
            result=audit_annotations(roots=args.root,files=args.file,policy_path=args.policy or DEFAULT_POLICY,output=args.out)
        elif args.action=="non-live-scan":
            from core.non_live_text import DEFAULT_POLICY,scan_metadata
            result=scan_metadata(input_path=args.input,policy_path=args.policy or DEFAULT_POLICY,output=args.out)
        elif args.action=="topic-distribution":
            from core.title_distribution import analyze_title_distribution
            result=analyze_title_distribution(args.input,args.out,topics=args.topics,
                sample_size=args.sample_size,batch_size=args.batch_size,seed=args.seed,
                max_features=args.max_features,encoder=args.encoder)
        elif args.action=="compare":
            from core.text_benchmark import compare_text
            result=compare_text(args.gold_dir,args.category,args.out,methods=args.methods,encoder=args.encoder,
                fields=args.fields,target=args.target,min_n=args.min_calibration_n)
        elif args.action=="release":
            from core.metadata_acceptance import release_round
            result=release_round(args.round,args.out)
        elif args.action=="next":
            from core.metadata_acceptance import inspect_round
            result=inspect_round(args.round)
        elif args.action=="evaluate":
            result=evaluate(args.round,args.labels,args.target)
        elif args.action=="train":
            result=train_topic(args.train,args.calibration,args.policy,args.output,args.encoder)
        elif args.action=="import-gold":
            result=import_metadata_gold(args.source_dir,args.category,args.out,args.seed)
        elif args.action=="benchmark":
            result=benchmark(args.model,args.test,args.out)
        elif args.action=="revise":
            result=revise_policy(args.policy,args.proposals,args.labels,args.select,args.output,args.validation_gold)
        elif args.action=="close":
            result=close_loop(args.rounds,args.output)
        elif args.action=="verify":
            from core.topic_verifier import verify_titles
            result=verify_titles(args.input,args.gold,args.policy,args.out,args.model,args.limit)
        else:
            result=propose(args.labels,args.policy,args.output,args.validation_gold)
        print(json.dumps(result,ensure_ascii=False,indent=2))
        if args.action=="doctor" and result["errors"]: return 2
        if args.action=="status" and result["status"].startswith("invalid"): return 2
        if args.action=="profile" and result["status"]!="execution_checks_passed": return 2
        if args.action=="evaluate" and result["status"] not in ("accepted","retrospective_replay_not_acceptance"): return 2
        return 0
    except (ValueError,OSError,KeyError) as e:
        print("[ERROR]",e,file=sys.stderr)
        return 1
if __name__=="__main__":
    raise SystemExit(main())
