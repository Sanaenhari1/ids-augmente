import json, datetime as dt
def log_event(kind, payload):
    with open("logs/audit.jsonl", "a") as f:
        f.write(json.dumps({"ts": dt.datetime.now().isoformat(), "kind": kind, **payload},
                           ensure_ascii=False, default=str) + "\n")
