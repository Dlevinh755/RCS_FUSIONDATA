import torch
import argparse
import pandas as pd
from pathlib import Path
from train_mlp import trainmlp


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.data_path:
        df = pd.read_csv(args.data_path + "/full_data.csv",low_memory=False)
        print(f"Data loaded from {args.data_path}, shape: {df.shape}")
        print(df.columns)
        
        # Tạo đường dẫn tuyệt đối cho file ảnh
        df["file_path"] = df["asin"].apply(lambda x: str(args.data_path + "/images/" + f"{x}.jpg"))
    else:
        df = None
    model, metrics = trainmlp(df, batch_size=args.batch_size, lr=args.lr, epochs=args.epochs, patience=args.patience, heads=args.heads, device=device)
    model_path = "mlp_camrec_model.pth"
    torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--data_path", type=str, default=None, help="Path to the directory containing train.csv and images/")
    args = parser.parse_args()
    main(args)