import json, os, pandas as pd, streamlit as st
st.set_page_config(page_title="IDS-Augmenté", layout="wide")
R = [json.loads(l) for l in open("reports/results.jsonl")]
st.title("IDS-Augmenté : Transformer + LLM local + Agent")
t1, t2, t3 = st.tabs(["Alertes", "Détail", "Métriques"])
with t1:
    st.dataframe(pd.DataFrame([{"id": r["alert"]["alert_id"], "classe": r["alert"]["predicted_class"],
        "confiance": r["alert"]["confidence"], "src": r["alert"]["src_ip"], "dst": r["alert"]["dst_ip"],
        "risque LLM": r["explanation"]["risk_level"], "verdict agent": r["agent"]["verdict"],
        "vérité terrain": r["truth"]} for r in R]))
with t2:
    sel = st.selectbox("Alerte", [r["alert"]["alert_id"] for r in R])
    r = next(x for x in R if x["alert"]["alert_id"] == sel)
    c1, c2 = st.columns(2)
    c1.subheader("Explication LLM"); c1.json(r["explanation"])
    c2.subheader("Investigation de l'agent"); c2.json(r["agent"])
    st.subheader("Alerte brute"); st.json(r["alert"])
with t3:
    for f in ("ft_metrics", "xgb_metrics"):
        if os.path.exists(f"reports/{f}.json"): st.json(json.load(open(f"reports/{f}.json")))
    if os.path.exists("reports/confusion_matrix.png"): st.image("reports/confusion_matrix.png")
