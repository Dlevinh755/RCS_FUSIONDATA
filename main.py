import torch
import argparse
import pandas as pd
from train_mlp import trainmlp


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.data_path:
        df = pd.read_csv(args.data_path)
        df["file_path"] = args.data_path + "images/" + df["asin"] + ".jpg"
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
    parser.add_argument("--data_path", type=str, default=None, help="Path to the CSV file containing the dataset")
    args = parser.parse_args()
    main(args)