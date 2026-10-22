# IDS-Augmenté : FT-Transformer + LLM local (Ollama) + agent d'investigation

Pipeline 100 % local : flux réseau -> FT-Transformer (détection) -> LLM (explication, MITRE ATT&CK)
-> agent à outils en lecture seule (investigation) -> verdict + journal d'audit.

## Résultats (UNSW-NB15, partition de test officielle, 82 332 flux)
| Modèle | macro-F1 | Taux de détection | Taux de faux positifs |
|---|---|---|---|
| FT-Transformer | 0,422 | 99,82 % | 42,1 % |
| XGBoost (baseline) | 0,514 | 99,39 % | 38,0 % |

Détection : 0,04 ms par flux sur CPU (6 cœurs, VM).
## Couche agent : 16 alertes (10 vraies attaques, 6 fausses alarmes du Transformer)
| Configuration | Exactitude |
|---|---|
| LLM seul (Llama 3.2 3B, sans outils) | 62,5 % (égal au taux de base : 10 alertes sur 16 sont de vraies attaques) |
| Agent, verdict brut du LLM | 68,8 % |
| Agent + garde-fou déterministe | 100 % |

- Le verdict brut de l'agent a écarté à tort 5 attaques sur 10 ; le garde-fou (IP source en blocklist) les a toutes corrigées.
- L'agent a écarté 6 fausses alarmes sur 6.
- Latence moyenne sur CPU : explication 23 s, investigation 153 s (l'exigence initiale de 10 s n'est pas atteinte sans GPU).
- Lecture prudente : blocklist et logs sont simulés à partir de la vérité terrain, et n = 16. Avec 0 attaque manquée sur 10, la borne haute à 95 % du taux d'attaques manquées est d'environ 26 %. Ces chiffres valident l'architecture, pas une performance en production.

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

## Données
Le dataset n'est pas inclus. Télécharge `UNSW_NB15_training-set.csv` et `UNSW_NB15_testing-set.csv`
(UNSW Canberra / Kaggle, voir leurs conditions d'utilisation) et place-les dans `data/raw/`.
Les modèles entraînés ne sont pas versionnés : `python src/train.py` les régénère dans `models/`.
Testé avec Python 3.12 et Ollama (modèle `llama3.2:3b`).
