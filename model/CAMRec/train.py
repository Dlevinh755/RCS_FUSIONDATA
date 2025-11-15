import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torch.cuda.amp import autocast, GradScaler
import gc

from model.CAMRec.model import CAMRec


def train(train_dl, val_dl, test_dl, users, items, full_pivot, 
          batch_size=16, lr=1e-3, epochs=50, patience=5, heads=4, device='cuda'):
    
    # Initialize model
    model = CAMRec(
        n_users=len(users),
        n_items=len(items),
        user_dim=128,
        item_dim=128,
        proj_dim=256,
        heads=heads,
        history_dim=full_pivot.shape[1] if full_pivot is not None else None,
    ).to(device)
    
    # Only optimize trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    scaler = GradScaler()
    
    # Training state
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    
    print(f"🚀 Training CAMRec | Trainable params: {sum(p.numel() for p in trainable_params):,}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            for k in batch:
                batch[k] = batch[k].to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            with autocast():
                predictions = model(batch)
                loss = loss_fn(predictions, batch['rating'])
            
            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            num_batches += 1
            
            # Update progress
            progress_bar.set_postfix({'Loss': f"{loss.item():.4f}"})
            
            # 👈 Clear batch from GPU memory
            del batch, predictions, loss
            
            # 👈 Periodic memory cleanup
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()
        
        avg_train_loss = train_loss / max(1, num_batches)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_dl:
                for k in batch:
                    batch[k] = batch[k].to(device, non_blocking=True)
                
                with autocast():
                    predictions = model(batch)
                    loss = loss_fn(predictions, batch['rating'])
                
                val_loss += loss.item()
                val_batches += 1
                
                # Clear validation batch
                del batch, predictions, loss
        
        avg_val_loss = val_loss / max(1, val_batches)
        
        # Early stopping logic
        if avg_val_loss < best_val_loss - 1e-6:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best state to CPU to free GPU memory
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")
        
        # 👈 Aggressive memory cleanup after each epoch
        torch.cuda.empty_cache()
        gc.collect()
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    
    # Test evaluation
    model.eval()
    test_mae, test_rmse = evaluate_model(model, test_dl, device)
    
    print(f"✅ Training complete! Test MAE={test_mae:.4f}, RMSE={test_rmse:.4f}")
    
    # Final cleanup
    torch.cuda.empty_cache()
    gc.collect()
    
    return model, (test_mae, test_rmse)

def evaluate_model(model, dataloader, device):
    """Memory-efficient evaluation"""
    abs_errors = []
    sq_errors = []
    
    with torch.no_grad():
        for batch in dataloader:
            for k in batch:
                batch[k] = batch[k].to(device, non_blocking=True)
            
            with autocast():
                predictions = model(batch)
            
            targets = batch['rating']
            abs_errors.append(torch.abs(predictions - targets).cpu())
            sq_errors.append((predictions - targets).pow(2).cpu())
            
            # Clear batch
            del batch, predictions, targets
    
    mae = torch.cat(abs_errors).mean().item()
    rmse = torch.sqrt(torch.cat(sq_errors).mean()).item()
    
    return mae, rmse
