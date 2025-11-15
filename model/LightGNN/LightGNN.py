import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.utils.data import DataLoader, Dataset
from text_model.roberta_model import TextEncoder
from image_model.vgg19_model import ImageEncoder
from datahelper import AmazonReviewDataset
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer

# -----------------------------
# 1) Simple multimodal encoders
# -----------------------------
class MetaEncoder(nn.Module):
    def __init__(self, in_dim=16, out_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )
    def forward(self, meta_feat):
        return self.net(meta_feat)  # [B, out_dim]



class MultimodalFusion(nn.Module):
    def __init__(self, img_dim, txt_dim, meta_dim, fused_dim=128):
        super().__init__()
        self.fused_dim = fused_dim
        # simple concat + MLP fusion
        self.mlp = nn.Sequential(
            nn.Linear(img_dim + txt_dim + meta_dim, 256),
            nn.ReLU(),
            nn.Linear(256, fused_dim)
        )
    def forward(self, e_img, e_txt, e_meta):
        # e_*: [N_items, dim]
        x = torch.cat([e_img, e_txt, e_meta], dim=-1)
        return self.mlp(x)  # [N_items, fused_dim]


# ---------------------------------
# 3) LightGCN implementation
# ---------------------------------
class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, emb_dim, n_layers=2, edge_index=None):
        """
        edge_index: tuple (user_indices, item_indices) for interactions (both 1D tensors)
        We'll build normalized adjacency from it.
        """
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_nodes = n_users + n_items
        self.emb_dim = emb_dim
        self.n_layers = n_layers

        # initial id embeddings for users (items will be replaced by multimodal fused embeddings)
        self.user_emb = nn.Embedding(n_users, emb_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)

        # item_emb placeholder (we'll set externally from multimodal fusion)
        self.item_emb_param = nn.Parameter(torch.zeros(n_items, emb_dim))
        # initialize tensor data explicitly (Parameter is a Tensor subclass; use .data for clarity)
        nn.init.xavier_uniform_(self.item_emb_param.data)
        # note: item_emb_param is kept for compatibility but not used in demo (we provide fused embeddings)

        # adjacency
        if edge_index is None:
            raise ValueError("edge_index required to build adjacency")
        self.register_buffer("edge_u", edge_index[0].long())
        self.register_buffer("edge_i", edge_index[1].long())
        self._build_norm_adj()

    def _build_norm_adj(self):
        # Build symmetric bipartite adjacency matrix in sparse form
        device = self.edge_u.device
        n = self.n_nodes
        # edges: u -> item (index shift items by n_users)
        rows = torch.cat([self.edge_u, self.edge_i + self.n_users], dim=0)
        cols = torch.cat([self.edge_i + self.n_users, self.edge_u], dim=0)
        vals = torch.ones(rows.size(0), device=device)

        # degree
        deg = torch.zeros(n, device=device).scatter_add_(0, rows, vals)
        deg = deg + 1e-12
        deg_inv_sqrt = deg.pow(-0.5)

        # store for propagation: we will compute normalized values on the fly
        self.register_buffer("adj_row", rows)
        self.register_buffer("adj_col", cols)
        self.register_buffer("adj_vals", vals)
        self.register_buffer("deg_inv_sqrt", deg_inv_sqrt)

    def propagate(self, user_feats, item_feats):
        """
        user_feats: [n_users, d]
        item_feats: [n_items, d]
        returns final_user_emb, final_item_emb after K-layer mean propagation and averaging layers
        """
        device = user_feats.device
        all_emb = torch.cat([user_feats, item_feats], dim=0)  # [n_nodes, d]
        embeddings_per_layer = [all_emb]

        for _ in range(self.n_layers):
            # message passing: y_i = sum_j (1/sqrt(deg_i) * 1/sqrt(deg_j)) * emb_j
            src = self.adj_row
            dst = self.adj_col
            vals = self.deg_inv_sqrt[src] * self.deg_inv_sqrt[dst]  # normalization per edge

            # aggregate: for each dst node sum vals * emb[src]
            msgs = all_emb[src] * vals.unsqueeze(-1)  # [n_edges, d]
            agg = torch.zeros_like(all_emb)
            agg = agg.index_add(0, dst, msgs)  # sum into dst positions

            all_emb = agg
            embeddings_per_layer.append(all_emb)

        # average embeddings across layers
        stacked = torch.stack(embeddings_per_layer, dim=0)  # [K+1, n_nodes, d]
        final = stacked.mean(dim=0)  # [n_nodes, d]
        final_user = final[:self.n_users]
        final_item = final[self.n_users:]
        return final_user, final_item

    def forward(self, fused_item_emb):
        """
        fused_item_emb: [n_items, emb_dim] — provided by multimodal fusion
        """
        # initial embeddings: user id emb, item emb replaced by fused
        u0 = self.user_emb.weight  # [n_users, d]
        i0 = fused_item_emb        # [n_items, d] (should match emb_dim)
        final_u, final_i = self.propagate(u0, i0)
        return final_u, final_i


# -----------------------------
# 4) BPR loss and dataset
# -----------------------------
class InteractionDataset(Dataset):
    def __init__(self, user_item_pairs, n_users, n_items, neg_sampler=True):
        # user_item_pairs: list of (u, i)
        self.pos = user_item_pairs
        self.n_users = n_users
        self.n_items = n_items

    def __len__(self):
        return len(self.pos)

    def __getitem__(self, idx):
        u, i = self.pos[idx]
        # sample negative (avoid sampling the positive item)
        j = random.randrange(self.n_items)
        while j == i:
            j = random.randrange(self.n_items)
        return torch.tensor(u, dtype=torch.long), torch.tensor(i, dtype=torch.long), torch.tensor(j, dtype=torch.long)

def bpr_loss(u_emb, i_emb, j_emb):
    # u_emb: [B, d], i_emb: [B, d], j_emb: [B, d]
    x_pos = (u_emb * i_emb).sum(dim=-1)
    x_neg = (u_emb * j_emb).sum(dim=-1)
    loss = -F.logsigmoid(x_pos - x_neg).mean()
    return loss

# -----------------------------
# 5) Putting all together: demo training
# -----------------------------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # reproducibility
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # --- Load Amazon review data (prefer full_data.csv, fallback to train.csv) ---
    repo_root = Path(__file__).resolve().parent.parent
    candidates = [
        repo_root / "data" / "amazonproduct" / "full_data.csv",
        repo_root / "data" / "amazonproduct" / "train.csv",
        repo_root / "data" / "amazon_product" / "full_data.csv",
    ]
    data_path = None
    for p in candidates:
        if p.exists():
            data_path = p
            break
    if data_path is None:
        raise FileNotFoundError("Cannot find Amazon dataset CSV in data/amazonproduct/")

    df = pd.read_csv(data_path)
    df['reviewerID'] = df['reviewerID'].astype(str)
    df['asin'] = df['asin'].astype(str)

    # create mappings and interactions from real data
    users = {u: i for i, u in enumerate(df['reviewerID'].unique())}
    items = {a: i for i, a in enumerate(df['asin'].unique())}
    n_users = len(users)
    n_items = len(items)
    interactions = [(users[r], items[a]) for _, (r, a) in df[['reviewerID', 'asin']].iterrows()]

    # build edge index
    edge_u = torch.tensor([u for (u, i) in interactions], dtype=torch.long, device=device)
    edge_i = torch.tensor([i for (u, i) in interactions], dtype=torch.long, device=device)

    # pivot table for history (passed into AmazonReviewDataset)
    pivot_df = pd.pivot_table(df, values='overall', index='reviewerID', columns='asin', aggfunc='mean', fill_value=0)

    # tokenizer for dataset (AmazonReviewDataset expects a tokenizer)
    tok = AutoTokenizer.from_pretrained('roberta-base')

    # Amazon dataset wrapper (used for other purposes if needed)
    amazon_ds = AmazonReviewDataset(df, users, items, tok, history_dim=pivot_df, max_len=128)

    # instantiate encoders and fusion (features still computed/cached as in demo)
    emb_dim = 64
    img_enc = ImageEncoder().to(device)
    txt_enc = TextEncoder('roberta-base').to(device)
    meta_enc = MetaEncoder(in_dim=16, out_dim=32).to(device)
    fusion = MultimodalFusion(img_dim=64, txt_dim=64, meta_dim=32, fused_dim=emb_dim).to(device)

    # LightGCN model with real sizes
    lightgcn = LightGCN(n_users=n_users, n_items=n_items, emb_dim=emb_dim, n_layers=2,
                        edge_index=(edge_u, edge_i)).to(device)

    optimizer = torch.optim.Adam(list(img_enc.parameters()) +
                                 list(txt_enc.parameters()) +
                                 list(meta_enc.parameters()) +
                                 list(fusion.parameters()) +
                                 list(lightgcn.user_emb.parameters()), lr=1e-3)

    # Interaction dataset (wrap real interactions)
    dataset = InteractionDataset(interactions, n_users, n_items)
    loader = DataLoader(dataset, batch_size=512, shuffle=True, drop_last=False)

    # For demo speed we use random item-level features (or you can precompute real features)
    img_feats = torch.randn(n_items, 2048, device=device)
    txt_feats = torch.randn(n_items, 768, device=device)
    meta_feats = torch.randn(n_items, 16, device=device)

    # training loop (same as before)
    epochs = 5
    for epoch in range(epochs):
        lightgcn.train(); img_enc.train(); txt_enc.train(); meta_enc.train(); fusion.train()
        total_loss = 0.0
        for batch in loader:
            u, i, j = [x.to(device) for x in batch]  # [B]
            # compute fused item embeddings for all items (can be cached if static)
            e_img = img_enc(img_feats)   # [n_items, feat_dim]
            e_txt = txt_enc(txt_feats)   # [n_items, feat_dim]
            e_meta = meta_enc(meta_feats) # [n_items, 32]
            fused_items = fusion(e_img, e_txt, e_meta)  # [n_items, emb_dim]

            # forward through LightGCN to get final node embeddings
            final_u, final_i = lightgcn(fused_items)  # [n_users, d], [n_items, d]

            # gather batch embeddings
            u_emb = final_u[u]   # [B, d]
            i_emb = final_i[i]   # [B, d]
            j_emb = final_i[j]   # [B, d]

            loss = bpr_loss(u_emb, i_emb, j_emb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * u.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{epochs}, avg BPR loss: {avg_loss:.4f}")

    # after training: final embeddings
    with torch.no_grad():
        e_img = img_enc(img_feats)
        e_txt = txt_enc(txt_feats)
        e_meta = meta_enc(meta_feats)
        fused_items = fusion(e_img, e_txt, e_meta)
        final_u, final_i = lightgcn(fused_items)
    print("Training done. Sample score user0-item0:", (final_u[0] * final_i[0]).sum().item())
