import torch.nn as nn



class CoAttentionBlock(nn.Module):
    def __init__(self, dim=256, num_heads=4, ffn_ratio=4):
        super().__init__()
        self.attn_text = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.attn_image = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.ln_t1 = nn.LayerNorm(dim); self.ln_i1 = nn.LayerNorm(dim)
        self.ffn_t = nn.Sequential(nn.Linear(dim, dim*ffn_ratio), nn.ReLU(), nn.Linear(dim*ffn_ratio, dim))
        self.ffn_i = nn.Sequential(nn.Linear(dim, dim*ffn_ratio), nn.ReLU(), nn.Linear(dim*ffn_ratio, dim))
        self.ln_t2 = nn.LayerNorm(dim); self.ln_i2 = nn.LayerNorm(dim)

    def forward(self, T, I):
        Tq = T.unsqueeze(1); Iq = I.unsqueeze(1)
        Tco, _ = self.attn_text(query=Tq, key=Iq, value=Iq)
        Ico, _ = self.attn_image(query=Iq, key=Tq, value=Tq)
        T1 = self.ln_t1((Tco + Tq).squeeze(1))
        I1 = self.ln_i1((Ico + Iq).squeeze(1))
        T2 = self.ln_t2(self.ffn_t(T1) + T1)
        I2 = self.ln_i2(self.ffn_i(I1) + I1)
        F = T2 * I2
        return T2, I2, F
    




class CrossAttentionBlock(nn.Module):

    def __init__(self, dim=256, num_heads=4, ffn_ratio=4, direction="text_to_image"):
        
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.direction = direction  # xác định chiều attention

        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)

        # Feed Forward Network (giống Transformer)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_ratio),
            nn.ReLU(),
            nn.Linear(dim * ffn_ratio, dim)
        )
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, T, I):
    
        # Thêm chiều sequence = 1
        Tq = T.unsqueeze(1)   # [B, 1, D]
        Iq = I.unsqueeze(1)   # [B, 1, D]

        if self.direction == "text_to_image":
            # text nhìn ảnh
            out, _ = self.attn(query=Tq, key=Iq, value=Iq)
            T1 = self.ln1((out + Tq).squeeze(1))
            T2 = self.ln2(self.ffn(T1) + T1)
            F = T2 * I  # kết hợp kiểu element-wise
            return T2, I, F

        elif self.direction == "image_to_text":
            # ảnh nhìn text
            out, _ = self.attn(query=Iq, key=Tq, value=Tq)
            I1 = self.ln1((out + Iq).squeeze(1))
            I2 = self.ln2(self.ffn(I1) + I1)
            F = I2 * T  # kết hợp kiểu element-wise
            return T, I2, F

        else:
            raise ValueError("direction must be 'text_to_image' or 'image_to_text'")
