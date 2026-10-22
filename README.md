# IDS-Augmenté : FT-Transformer + LLM local (Ollama) + agent d'investigation

Pipeline 100 % local : flux réseau -> FT-Transformer (détection) -> LLM (explication, MITRE ATT&CK)
-> agent à outils en lecture seule (investigation) -> verdict + journal d'audit.

## Résultats (UNSW-NB15, partition de test officielle, 82 332 flux)
| Modèle | macro-F1 | Taux de détection | Taux de faux positifs |
|---|---|---|---|
| FT-Transformer | 0,422 | 99,82 % | 42,1 % |
| XGBoost (baseline) | 0,514 | 99,39 % | 38,0 % |

Détection : 0,04 ms par flux sur CPU (6 cœurs, VM).
Agent vs LLM seul, sur X alertes : voir `python src/evaluate.py` (X % vs X %).

## Choix de conception
- Le Transformer décide, le LLM explique et investigue ; il n'est jamais l'unique décideur.
- Sorties LLM contraintes par JSON Schema ; techniques MITRE validées contre une liste blanche.
- Agent : outils en lecture seule, liste blanche, logs traités comme données non fiables.
- Garde-fou déterministe : une IP source en blocklist ne peut pas être classée fausse alerte.
  Le verdict brut du LLM et le verdict final sont tous deux journalisés.

## Limites (à lire avant d'interpréter les chiffres)
- UNSW-NB15 est un dataset de laboratoire ; un taux de faux positifs de 42 % n'est pas exploitable
  seul en production, d'où la couche d'investigation.
- Les adresses IP et les logs de l'agent sont SIMULÉS, et générés à partir de la vérité terrain
  (une attaque produit des logs d'attaque). L'exactitude de l'agent mesure donc sa capacité à lire
  un contexte cohérent, pas sa performance sur un réseau réel.
- Petit échantillon d'alertes pour l'agent (latence LLM de 1 à 9 minutes par alerte sur CPU).
- Modèle LLM de 3 milliards de paramètres : rapports parfois incohérents avec le verdict.
- Pas de validation sur un second dataset.

## Lancer
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cpu && uv pip install -r requirements.txt
python src/train.py && python src/run_pipeline.py && python src/evaluate.py && streamlit run src/app.py
