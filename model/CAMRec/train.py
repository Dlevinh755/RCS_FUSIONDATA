import torch
import torch.nn as nn
from tqdm.auto import tqdm

from model.CAMRec.model import CAMRec


def train(train_dl, val_dl, test_dl, users, items, full_pivot ,batch_size=16, lr=1e-3, epochs=50, patience=5, heads=4, device='cuda'):
    best_val = float('inf') 
    best_state = None 
    bad = 0 

    model = CAMRec(
        n_users=len(users),
        n_items=len(items),
        user_dim=128,
        item_dim=128,
        proj_dim=256,
        heads=heads,
        history_dim=full_pivot.shape[1],
    ).to(device)
    model.text_enc.eval()
    model.img_enc.eval()
    optim = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    loss_fn = nn.MSELoss()
    
    for ep in range(epochs):
        model.train()
        tot = 0.0; n = 0
        progress_bar = tqdm(train_dl, desc=f"Epoch {ep+1}/{epochs}", leave=False)
        
        for batch in progress_bar:
            for k in batch:
                batch[k] = batch[k].to(device)
    
            optim.zero_grad()
            yhat = model(batch)
            loss = loss_fn(yhat, batch['rating'])
            loss.backward()
            optim.step()
            
            batch_size_curr = len(yhat)
            tot += loss.item() * batch_size_curr
            n += batch_size_curr
            progress_bar.set_postfix({'Loss': loss.item()})
            
        train_loss = tot / max(1, n)

        # Validate
        model.eval()
        with torch.no_grad():
            tot = 0.0; n = 0
            for batch in val_dl:
                for k in batch: batch[k] = batch[k].to(device)
                yhat = model(batch)
                loss = loss_fn(yhat, batch['rating'])
                tot += loss.item() * len(yhat); n += len(yhat)
            val_loss = tot / max(1,n)

        print(f"[Epoch {ep+1}] train MSE={train_loss:.4f} | val MSE={val_loss:.4f}")
        if val_loss + 1e-6 < best_val:
            best_val = val_loss; bad = 0
            best_state = {k:v.cpu() for k,v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict({k:v.to(device) for k,v in best_state.items()})
    model.eval()
    abs_errs, sq_errs = [], []
    with torch.no_grad():
        for batch in test_dl:
            for k in batch:
                batch[k] = batch[k].to(device)
            yhat = model(batch)
            y = batch['rating']
            abs_errs.append(torch.abs(yhat - y))
            sq_errs.append((yhat - y) ** 2)
    mae = torch.cat(abs_errs).mean().item()
    rmse = torch.sqrt(torch.cat(sq_errs).mean()).item()
    print(f"Test MAE={mae:.4f} | RMSE={rmse:.4f}")
    return model, (mae, rmse)
