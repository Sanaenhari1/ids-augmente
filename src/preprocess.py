import numpy as np, pandas as pd, joblib
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split

CAT = ["proto", "service", "state"]
DROP = ["id", "label", "attack_cat"]
TRAIN = "data/raw/UNSW_NB15_training-set.csv"
TEST = "data/raw/UNSW_NB15_testing-set.csv"

def load_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["attack_cat"] = df["attack_cat"].astype(str).str.strip()
    return df

def transform(df, meta):
    xn = df[meta["num_cols"]].astype(float).fillna(0).copy()
    sk = meta["skewed"]
    xn[sk] = np.log1p(xn[sk].clip(lower=0))
    xn = meta["scaler"].transform(xn).astype("float32")
    xc = (meta["enc"].transform(df[CAT].astype(str)) + 1).astype("int64")  # 0 = inconnu
    return xn, xc

def prepare():
    tr, te = load_csv(TRAIN), load_csv(TEST)
    le = LabelEncoder().fit(tr["attack_cat"])
    y_tr, y_te = le.transform(tr["attack_cat"]), le.transform(te["attack_cat"])
    i_tr, i_va = train_test_split(np.arange(len(tr)), test_size=0.15, stratify=y_tr, random_state=42)
    fit_df = tr.iloc[i_tr]
    num_cols = [c for c in tr.columns if c not in CAT + DROP]
    skewed = [c for c in num_cols if fit_df[c].min() >= 0 and fit_df[c].skew() > 2]
    tmp = fit_df[num_cols].astype(float).copy()
    tmp[skewed] = np.log1p(tmp[skewed])
    scaler = StandardScaler().fit(tmp)
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(fit_df[CAT].astype(str))
    meta = {"scaler": scaler, "enc": enc, "num_cols": num_cols, "skewed": skewed,
            "classes": list(le.classes_), "cards": [len(c) for c in enc.categories_]}
    joblib.dump(meta, "models/preproc.joblib")
    out = {"meta": meta}
    for name, df, y, idx in (("train", tr, y_tr, i_tr), ("val", tr, y_tr, i_va), ("test", te, y_te, None)):
        d = df if idx is None else df.iloc[idx]
        xn, xc = transform(d, meta)
        out[name] = (xn, xc, y if idx is None else y[idx])
    return out
