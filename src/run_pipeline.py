import argparse, json, time, datetime as dt
import numpy as np, pandas as pd
from preprocess import load_csv, TEST
from detect import load_detector, predict_proba
import simulate as sim, agent
from explain import explain
from audit import log_event

ap = argparse.ArgumentParser()
ap.add_argument("--n-tp", type=int, default=8)
ap.add_argument("--n-fp", type=int, default=4)
a = ap.parse_args()
FLOW = ["proto", "service", "state", "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sttl"]

rng = np.random.default_rng(42)
df = load_csv(TEST)
model, meta = load_detector(); classes = meta["classes"]; normal = classes.index("Normal")
t0 = time.time(); P = predict_proba(model, meta, df); pred = P.argmax(1)
print(f"Détection de {len(df)} flux en {time.time()-t0:.1f}s ({(time.time()-t0)/len(df)*1000:.3f} ms/flux)")

is_att = (df["attack_cat"] != "Normal").values
cand = np.where(pred != normal)[0]
tp_idx = [i for i in cand if is_att[i]]; fp_idx = [i for i in cand if not is_att[i]]
sel = list(rng.choice(tp_idx, min(a.n_tp, len(tp_idx)), replace=False)) + \
      list(rng.choice(fp_idx, min(a.n_fp, len(fp_idx)), replace=False))
rng.shuffle(sel)

now, items, events = dt.datetime.now(), [], []
for k, i in enumerate(sel):
    row = df.iloc[i]; att = bool(is_att[i])
    src, dst = sim.assign(att, rng)
    ts = now - dt.timedelta(minutes=int(rng.integers(1, 60)))
    flow = {c: (row[c].item() if hasattr(row[c], "item") else row[c]) for c in FLOW}
    alert = {"alert_id": f"ALT-{k:04d}", "timestamp": ts.isoformat(timespec="seconds"), "src_ip": src,
             "dst_ip": dst, "predicted_class": classes[pred[i]], "confidence": round(float(P[i, pred[i]]), 4),
             "flow": flow}
    events += sim.make_events(ts, src, dst, att, row["attack_cat"], rng)
    items.append({"alert": alert, "truth": "true_positive" if att else "false_positive",
                  "true_class": row["attack_cat"]})

pd.DataFrame(events).sort_values("timestamp").to_csv("data/raw/events.csv", index=False)
open("data/raw/blocklist.txt", "w").write("\n".join(sim.BLOCKLIST))
agent.init("data/raw/events.csv", "data/raw/blocklist.txt", [x["alert"] for x in items])

with open("reports/results.jsonl", "w") as out:
    for n, x in enumerate(items, 1):
        al = x["alert"]; log_event("detection", al)
        t = time.time(); ex = explain(al); t_ex = time.time() - t
        log_event("llm_explanation", {"alert_id": al["alert_id"], "explanation": ex, "latency_s": t_ex})
        t = time.time(); ag = agent.investigate(al); t_ag = time.time() - t
        log_event("agent_verdict", {"alert_id": al["alert_id"], "verdict": ag.model_dump(), "latency_s": t_ag})
        t = time.time(); lo = agent.llm_only(al); t_lo = time.time() - t
        rec = {**x, "explanation": ex, "agent": ag.model_dump(), "llm_only": lo.model_dump(),
               "latency": {"explain": t_ex, "agent": t_ag, "llm_only": t_lo}}
        out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()
        print(f"[{n}/{len(items)}] {al['alert_id']} vérité={x['truth']} agent={ag.verdict} llm_seul={lo.verdict} ({t_ex+t_ag+t_lo:.0f}s)", flush=True)
