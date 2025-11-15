from typing import List
import torch
import torch.nn as nn
from attentionblock import CoAttentionBlock
from text_model.roberta_model import TextEncoder
from image_model.vgg19_model import ImageEncoder

class CAMRec(nn.Module):
    def __init__(self, n_users, n_items, user_dim=128, item_dim=128, 
                 proj_dim=256, heads=4, history_dim=None):
        super().__init__()
        
        # Core embeddings
        self.user_emb = nn.Embedding(n_users, user_dim)
        self.item_emb = nn.Embedding(n_items, item_dim)
        
        # User-Item MLP
        self.ui_mlp = nn.Sequential(
            nn.Linear(user_dim + item_dim, 256),
            nn.LeakyReLU(inplace=True),  # 👈 inplace=True saves memory
            nn.Linear(256, 256),
            nn.LeakyReLU(inplace=True),
        )
        
        # Multimodal encoders (frozen for memory efficiency)
        self.text_enc = TextEncoder('roberta-base')
        self.img_enc = ImageEncoder()
        
        # Freeze encoders to save memory
        for param in self.text_enc.parameters():
            param.requires_grad = False
        for param in self.img_enc.parameters():
            param.requires_grad = False
            
        # Set to eval mode permanently
        self.text_enc.eval()
        self.img_enc.eval()
        
        # Projection layers
        self.text_proj = nn.Linear(768, proj_dim)
        self.img_proj = nn.Linear(4096, proj_dim)

        # Optional: project history-derived preference into 128-d
        self.hist_proj = nn.Sequential(
            nn.Linear(item_dim, 128),
            nn.LeakyReLU(inplace=True)
        )

        # Cross-attention
        self.coattn = CoAttentionBlock(dim=proj_dim, num_heads=heads, ffn_ratio=4)

        # Predictor: add +128 when history is present
        self.pred_mlp = nn.Sequential(
            nn.Linear(256 + proj_dim + 128, 256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

    def forward(self, batch):
        # Extract features with no gradient computation
        with torch.no_grad():
            cls = self.text_enc(batch['input_ids'], batch['attention_mask'])
            img_features = self.img_enc(batch['image'])

        user_emb = self.user_emb(batch['user_idx'])
        item_emb = self.item_emb(batch['item_idx'])

        ui_interaction = self.ui_mlp(torch.cat([user_emb, item_emb], dim=1))

        text_proj = self.text_proj(cls)
        img_proj = self.img_proj(img_features)

        _, _, fused_features = self.coattn(text_proj, img_proj)

        # History preference embedding (weighted sum of item embeddings)
        if 'historical_ratings' in batch:
            # [B, n_items] @ [n_items, item_dim] -> [B, item_dim]
            pref_vec = batch['historical_ratings'] @ self.item_emb.weight.detach()
            hist_pref = self.hist_proj(pref_vec)
        else:
            hist_pref = torch.zeros(user_emb.size(0), 128, device=user_emb.device)

        combined = torch.cat([ui_interaction, fused_features, hist_pref], dim=1)
        rating = self.pred_mlp(combined).squeeze(1)

        # 👈 Clear intermediate tensors to free memory
        del cls, img_features, text_proj, img_proj, combined
        
        return rating

def collate_fn(batch: List[dict]):
    """Memory-efficient collate function"""
    out = {
        'user_idx': torch.stack([b['user_idx'] for b in batch]),
        'item_idx': torch.stack([b['item_idx'] for b in batch]),
        'input_ids': torch.stack([b['input_ids'] for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
        'image': torch.stack([b['image'] for b in batch]),
        'rating': torch.stack([b['rating'] for b in batch]),
    }
    if 'historical_ratings' in batch[0]:
        out['historical_ratings'] = torch.stack([b['historical_ratings'] for b in batch])
    return out


