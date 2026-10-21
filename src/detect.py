import json, joblib, torch
from preprocess import transform
from model import FTTransformer

def load_detector():
    meta = joblib.load("models/preproc.joblib")
    arch = json.load(open("models/model_cfg.json"))["arch"]
    m = FTTransformer(len(meta["num_cols"]), meta["cards"], len(meta["classes"]), **arch)
    m.load_state_dict(torch.load("models/ft_transformer.pt", map_location="cpu")); m.eval()
    return m, meta

@torch.no_grad()
def predict_proba(m, meta, df, bs=4096):
    xn, xc = transform(df, meta)
    out = [torch.softmax(m(torch.tensor(xn[i:i+bs]), torch.tensor(xc[i:i+bs])), 1)
           for i in range(0, len(xn), bs)]
    return torch.cat(out).numpy()
