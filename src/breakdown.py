import json
BL = set(open("data/raw/blocklist.txt").read().split())
R = [json.loads(l) for l in open("reports/results.jsonl")]
print(f"{'id':9s}{'vérité':16s}{'src_ip':16s}{'blocklist':10s}{'LLM brut':16s}{'final':16s}{'LLM seul'}")
for r in R:
    a = r["alert"]
    print(f"{a['alert_id']:9s}{r['truth']:16s}{a['src_ip']:16s}{str(a['src_ip'] in BL):10s}"
          f"{r['agent']['raw_verdict']:16s}{r['agent']['verdict']:16s}{r['llm_only']['verdict']}")
att = [r for r in R if r["truth"] == "true_positive"]
for name, sel in (("en blocklist", [r for r in att if r["alert"]["src_ip"] in BL]),
                  ("hors blocklist", [r for r in att if r["alert"]["src_ip"] not in BL])):
    ok = sum(r["agent"]["raw_verdict"] == "true_positive" for r in sel)
    print(f"Attaques {name:15s}: {len(sel)} | LLM brut correct : {ok}/{len(sel)}")
print("Alertes avec le log piégé (203.0.113.10) :", sum(r["alert"]["src_ip"] == "203.0.113.10" for r in R))
