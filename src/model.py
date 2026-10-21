import torch, torch.nn as nn

class FTTransformer(nn.Module):
    def __init__(self, n_num, cat_cards, n_classes, d=32, heads=4, layers=2, drop=0.1):
        super().__init__()
        self.num_w = nn.Parameter(torch.randn(n_num, d) * 0.02)
        self.num_b = nn.Parameter(torch.zeros(n_num, d))
        self.cat_emb = nn.ModuleList([nn.Embedding(c + 1, d) for c in cat_cards])
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, d * 2, drop, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.ReLU(), nn.Linear(d, n_classes))

    def forward(self, x_num, x_cat):
        t_num = x_num.unsqueeze(-1) * self.num_w + self.num_b
        t_cat = torch.stack([e(x_cat[:, i]) for i, e in enumerate(self.cat_emb)], 1)
        cls = self.cls.expand(x_num.size(0), -1, -1)
        h = self.enc(torch.cat([cls, t_num, t_cat], dim=1))
        return self.head(h[:, 0])
