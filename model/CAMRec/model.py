from typing import List
import torch
import torch.nn as nn
from attentionblock import CoAttentionBlock
from text_model.roberta_model import TextEncoder
from image_model.vgg19_model import ImageEncoder
import torch.nn.functional as F

class CAMRec(nn.Module):
    def __init__(self, n_users, n_items,
                 user_dim=128, item_dim=128,
                 proj_dim=256, heads=4,
                 history_dim=None):
        super().__init__()
        # ===== phần gốc =====
        self.user_emb = nn.Embedding(n_users, user_dim)
        self.item_emb = nn.Embedding(n_items, item_dim)
        self.ui_mlp = nn.Sequential(
            nn.Linear(user_dim + item_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
        )
        self.text_enc = TextEncoder('roberta-base')
        self.img_enc = ImageEncoder()
        self.text_proj = nn.Linear(768, proj_dim)
        self.img_proj = nn.Linear(4096, proj_dim)
        self.coattn = CoAttentionBlock(dim=proj_dim, num_heads=heads, ffn_ratio=4)

        # ===== encoder cho history =====
        # history_dim chính là độ dài vector historical_ratings (số cột của pivot)
        fused_in_dim = 256 + proj_dim
        # self.history_enc = None
        # if history_dim is not None:
        #     self.history_enc = nn.Sequential(
        #         nn.Linear(history_dim, 256),
        #         nn.LeakyReLU(),
        #         nn.Linear(256, 128),
        #         nn.LeakyReLU(),
        #     )
        #     fused_in_dim += 128

        # ===== predictor =====
        self.pred_mlp = nn.Sequential(
            nn.Linear(fused_in_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, batch):
        # Disable gradient cho text và image encoders trong training
        with torch.no_grad():
            cls = self.text_enc(batch['input_ids'], batch['attention_mask'])
            I4096 = self.img_enc(batch['image'])
        
        # Chỉ compute gradient cho phần cần thiết
        pu = self.user_emb(batch['user_idx'])
        qi = self.item_emb(batch['item_idx'])
        E = self.ui_mlp(torch.cat([pu, qi], dim=1))
        
        T0 = self.text_proj(cls.detach())  # Detach để tránh gradient
        I0 = self.img_proj(I4096.detach())
        
        T, I, F = self.coattn(T0, I0)
        
        feats = [E, F]
        V = torch.cat(feats, dim=1)
        yhat = self.pred_mlp(V).squeeze(1)
        return yhat


# =========================
# 4) Collate & DataLoader
# =========================
def collate_fn(batch: List[dict]):
    out = {
        'user_idx': torch.stack([b['user_idx'] for b in batch]),
        'item_idx': torch.stack([b['item_idx'] for b in batch]),
        'input_ids': torch.stack([b['input_ids'] for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
        'image': torch.stack([b['image'] for b in batch]),
        'rating': torch.stack([b['rating'] for b in batch]),
        # history phải cùng chiều, Dataset của bạn đã trả về tensor 1D rồi
        'historical_ratings': torch.stack([b['historical_ratings'] for b in batch]),
    }
    return out


