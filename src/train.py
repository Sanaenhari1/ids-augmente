import argparse, json, os, time
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from preprocess import prepare
from model import FTTransformer
from metrics import binary_report

ap = argparse.ArgumentParser()
ap.add_argument("--d", type=int, default=32)
ap.add_argument("--layers", type=int, default=2)
ap.add_argument("--epochs", type=int, default=15)
ap.add_argument("--bs", type=int, default=1024)
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(os.cpu_count()); torch.manual_seed(42)
D = prepare(); meta = D["meta"]; classes = meta["classes"]; normal = classes.index("Normal")
print("device:", dev, "| classes:", classes)

def loader(split, shuffle):
    xn, xc, y = D[split]
    return DataLoader(TensorDataset(torch.tensor(xn), torch.tensor(xc), torch.tensor(y)),
                      batch_size=a.bs, shuffle=shuffle)

tr_l, va_l, te_l = loader("train", True), loader("val", False), loader("test", False)
arch = {"d": a.d, "heads": 4, "layers": a.layers, "drop": 0.1}
json.dump({"arch": arch}, open("models/model_cfg.json", "w"))
model = FTTransformer(D["train"][0].shape[1], meta["cards"], len(classes), **arch).to(dev)

w = compute_class_weight("balanced", classes=np.arange(len(classes)), y=D["train"][2])
loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32).to(dev))
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

def evaluate(l):
    model.eval(); P, Y = [], []
    with torch.no_grad():
        for xn, xc, y in l:
            P.append(model(xn.to(dev), xc.to(dev)).argmax(1).cpu()); Y.append(y)
    return torch.cat(Y).numpy(), torch.cat(P).numpy()

best, t0 = 0, time.time()
for ep in range(a.epochs):
    model.train(); tot = 0
    for xn, xc, y in tr_l:
        opt.zero_grad()
        loss = loss_fn(model(xn.to(dev), xc.to(dev)), y.to(dev))
        loss.backward(); opt.step(); tot += loss.item()
    y, p = evaluate(va_l); f1 = f1_score(y, p, average="macro")
    print(f"epoch {ep+1:02d} | loss {tot/len(tr_l):.3f} | val macro-F1 {f1:.4f} | {time.time()-t0:.0f}s", flush=True)
    if f1 > best:
        best = f1; torch.save(model.state_dict(), "models/ft_transformer.pt")

model.load_state_dict(torch.load("models/ft_transformer.pt", map_location=dev))
y, p = evaluate(te_l)
print(classification_report(y, p, target_names=classes, digits=4, zero_division=0))
b = binary_report(y, p, normal)
print("Binaire :", {k: round(v, 4) if isinstance(v, float) else v for k, v in b.items()})
json.dump({"model": "FT-Transformer", "macro_f1": float(f1_score(y, p, average="macro")),
           "accuracy": float((y == p).mean()), "binary": b, "train_seconds": time.time() - t0},
          open("reports/ft_metrics.json", "w"), indent=2)

cm = confusion_matrix(y, p, normalize="true")
fig, ax = plt.subplots(figsize=(8, 7)); ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right")
ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
ax.set_xlabel("Prédit"); ax.set_ylabel("Réel")
for i in range(len(classes)):
    for j in range(len(classes)):
        ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7)
plt.tight_layout(); plt.savefig("reports/confusion_matrix.png", dpi=150)
