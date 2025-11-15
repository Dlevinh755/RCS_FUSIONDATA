import argparse
import torch

from load_data import load_data
from model.__init__ import CAMRec_train, LightGNN_train


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_dl, val_dl, test_dl, users, items, full_pivot = load_data(
        data_path=args.data_path,
        data_type=args.dataset,
        batch_size=args.batch_size,
    )
    if args.model == "CAMRec":
        model, metrics = CAMRec_train(
            train_dl,
            val_dl,
            test_dl,
            users,
            items,
            full_pivot,
            batch_size=args.batch_size,
            lr=args.lr,
            epochs=args.epochs,
            patience=args.patience,
            heads=args.heads,
            device=device,
        )
    elif args.model == "LightGNN":
    
        model, metrics = LightGNN_train(
            train_dl,
            val_dl,
            test_dl,
            users,
            items,
            full_pivot,
            batch_size=args.batch_size,
            lr=args.lr,
            epochs=args.epochs,
            patience=args.patience,
            heads=args.heads,
            device=device,
        )

    else:
        raise ValueError(f"Model {args.model} not recognized.")

    print(f"Training complete. Test MAE={metrics[0]:.4f}, RMSE={metrics[1]:.4f}")
    model_path = "mlp_camrec_model.pth"
    torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--model", type=str, default="CAMRec")
    parser.add_argument("--dataset", type=str, default="amazonproduct")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--data_path", type=str, default=None, help="Path to the directory containing train.csv and images/")
    args = parser.parse_args()
    main(args)