import json, os, pandas as pd
R = [json.loads(l) for l in open("reports/results.jsonl")]
df = pd.DataFrame([{"truth": r["truth"], "agent": r["agent"]["verdict"],
                    "raw": r["agent"].get("raw_verdict") or r["agent"]["verdict"],
                    "llm": r["llm_only"]["verdict"],
                    "t_ex": r["latency"]["explain"], "t_ag": r["latency"]["agent"]} for r in R])
fp, tp = df[df.truth == "false_positive"], df[df.truth == "true_positive"]
print(f"Alertes évaluées            : {len(df)} ({len(tp)} vraies, {len(fp)} fausses alarmes du Transformer)")
print(f"Exactitude LLM SEUL         : {(df.llm == df.truth).mean():.1%}")
print(f"Exactitude AGENT (LLM brut) : {(df.raw == df.truth).mean():.1%}")
print(f"Exactitude AGENT + garde-fou: {(df.agent == df.truth).mean():.1%}")
print(f"Corrections du garde-fou    : {(df.raw != df.agent).sum()}")
print(f"Fausses alarmes écartées    : {(fp.agent == 'false_positive').mean():.1%}  (réduction du bruit)")
print(f"Attaques écartées à tort    : {(tp.agent == 'false_positive').mean():.1%}  (risque critique)")
print(f"Renvoyées à un humain       : {(df.agent == 'needs_human').mean():.1%}")
print(f"Latence explication / agent : {df.t_ex.mean():.0f}s / {df.t_ag.mean():.0f}s")
for f in ("ft_metrics", "xgb_metrics"):
    p = f"reports/{f}.json"
    if os.path.exists(p):
        m = json.load(open(p)); b = m["binary"]
        print(f"{m['model']:15s} macro-F1={m['macro_f1']:.4f}  détection={b['detection_rate']:.4f}  FPR={b['false_positive_rate']:.4f}")
