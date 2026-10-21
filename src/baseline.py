import json, time, numpy as np, xgboost as xgb
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight
from preprocess import prepare
from metrics import binary_report

D = prepare(); classes = D["meta"]["classes"]; normal = classes.index("Normal")
X = lambda s: np.hstack([D[s][0], D[s][1].astype("float32")])
t0 = time.time()
clf = xgb.XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1,
                        tree_method="hist", n_jobs=-1, eval_metric="mlogloss")
clf.fit(X("train"), D["train"][2], sample_weight=compute_sample_weight("balanced", D["train"][2]))
p, y = clf.predict(X("test")), D["test"][2]
print(classification_report(y, p, target_names=classes, digits=4, zero_division=0))
b = binary_report(y, p, normal)
print("Binaire :", {k: round(v, 4) if isinstance(v, float) else v for k, v in b.items()})
json.dump({"model": "XGBoost", "macro_f1": float(f1_score(y, p, average="macro")),
           "accuracy": float((y == p).mean()), "binary": b, "train_seconds": time.time() - t0},
          open("reports/xgb_metrics.json", "w"), indent=2)
