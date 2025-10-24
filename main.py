import torch
import argparse
import pandas as pd
from train_mlp import trainmlp


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    meta = pd.read_csv("/kaggle/input/amazon-product/meta.csv")
    review = pd.read_csv("/kaggle/input/amazon-product/reviews.csv")
    df = review.merge(meta, on="asin")
    df["file_path"] = "/kaggle/input/amazon-product/images/"+ df["asin"] +".jpg"
    model, metrics = trainmlp(df, batch_size=args.batch_size, lr=args.lr, epochs=args.epochs, patience=args.patience, heads=args.heads, device=device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--heads", type=int, default=4)
    args = parser.parse_args()
    main(args)