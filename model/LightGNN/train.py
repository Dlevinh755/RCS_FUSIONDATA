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
from model.LightGNN.model import LightGCN, MultimodalFusion, MetaEncoder, InteractionDataset

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
