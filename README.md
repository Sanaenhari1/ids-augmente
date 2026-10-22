# IDS-Augmenté : FT-Transformer + LLM local (Ollama) + agent d'investigation

Pipeline 100 % local : flux réseau -> FT-Transformer (détection) -> LLM (explication, MITRE ATT&CK)
-> agent à outils en lecture seule (investigation) -> rapport + journal d'audit.

## Résultats (à remplir avec TES chiffres : python src/evaluate.py)
| Modèle | macro-F1 | Taux de détection | FPR |
|---|---|---|---|
| FT-Transformer | X | X | X |
| XGBoost (baseline) | X | X | X |

Agent vs LLM seul : exactitude X % vs X %, fausses alarmes écartées X %.

## Choix de conception
- Le Transformer décide, le LLM explique : il ne décide jamais seul de la détection.
- Sorties LLM contraintes par JSON Schema ; techniques MITRE validées contre une liste blanche.
- Agent : outils en lecture seule, liste blanche, 4 étapes max, logs traités comme données non fiables
  (défense contre la prompt injection indirecte, testée avec un log piégé).

## Limites
- UNSW-NB15 est un dataset de laboratoire ; les adresses IP et les logs de l'agent sont simulés.
- Petit échantillon d'alertes pour l'agent (latence LLM sur CPU). Pas de validation sur un 2e dataset.

## Lancer
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python src/train.py && python src/run_pipeline.py && python src/evaluate.py && streamlit run src/app.py
