def binary_report(y, p, normal_idx):
    yt, yp = y != normal_idx, p != normal_idx
    tp, fn = int((yt & yp).sum()), int((yt & ~yp).sum())
    fp, tn = int((~yt & yp).sum()), int((~yt & ~yp).sum())
    return {"detection_rate": tp / (tp + fn), "false_positive_rate": fp / (fp + tn),
            "precision": tp / (tp + fp), "tp": tp, "fn": fn, "fp": fp, "tn": tn}
