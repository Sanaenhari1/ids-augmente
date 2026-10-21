import json, ollama
from typing import Literal
from pydantic import BaseModel, Field

MODEL = "llama3.2:3b"

class Explanation(BaseModel):
    summary: str = Field(description="Explication en 2-3 phrases")
    risk_level: Literal["low", "medium", "high", "critical"]
    mitre_technique_ids: list[str] = Field(description="IDs MITRE ATT&CK, ex: T1046")
    recommendations: list[str]

MITRE_OK = {"T1046": "Network Service Discovery", "T1595": "Active Scanning",
            "T1498": "Network Denial of Service", "T1499": "Endpoint Denial of Service",
            "T1190": "Exploit Public-Facing Application", "T1110": "Brute Force",
            "T1048": "Exfiltration Over Alternative Protocol", "T1059": "Command and Scripting Interpreter",
            "T1105": "Ingress Tool Transfer", "T1210": "Exploitation of Remote Services",
            "T1071": "Application Layer Protocol", "T1021": "Remote Services"}
CLASS_MITRE = {"Reconnaissance": ["T1595", "T1046"], "DoS": ["T1498", "T1499"], "Exploits": ["T1190"],
               "Fuzzers": ["T1190"], "Backdoor": ["T1105"], "Shellcode": ["T1059"],
               "Analysis": ["T1595"], "Worms": ["T1210"]}

SYSTEM = """Tu es un analyste SOC. Tu reçois UNE alerte d'un IDS au format JSON.
Base-toi UNIQUEMENT sur les données fournies, n'invente aucun fait. Si l'information est insuffisante, dis-le.
Réponds en français, en JSON conforme au schéma."""

def explain(alert, model=MODEL):
    try:
        r = ollama.chat(model=model,
                        messages=[{"role": "system", "content": SYSTEM},
                                  {"role": "user", "content": json.dumps(alert, ensure_ascii=False)}],
                        format=Explanation.model_json_schema(), options={"temperature": 0})
        out = Explanation.model_validate_json(r.message.content).model_dump()
    except Exception as e:
        out = {"summary": f"Explication indisponible ({type(e).__name__})", "risk_level": "medium",
               "mitre_technique_ids": [], "recommendations": []}
    ids = [t for t in out["mitre_technique_ids"] if t in MITRE_OK]       # anti-hallucination
    out["mitre_technique_ids"] = ids or CLASS_MITRE.get(alert["predicted_class"], [])
    out["mitre_names"] = {t: MITRE_OK[t] for t in out["mitre_technique_ids"]}
    return out
