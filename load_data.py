from pathlib import Path
from typing import Optional, Union

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from datahelper import AmazonReviewDataset, filter_valid_rows
from model.CAMRec.model import collate_fn

def _any_image_exists(paths: pd.Series) -> bool:
    """Return True if at least one path in the series points to an existing image file."""
    for raw_path in paths.dropna().astype(str).head(128):
        path = raw_path.strip()
        if not path:
            continue
        if Path(path).exists():
            return True
    return False


def _resolve_dataset_dir(data_path: Optional[Union[str, Path]]) -> Path:
    """Resolve the directory that contains train/val/test splits."""
    if data_path is not None:
        candidate = Path(data_path).expanduser().resolve()
        if candidate.is_file():
            if candidate.suffix.lower() != ".csv":
                raise FileNotFoundError(f"Provided file {candidate} is not a CSV dataset.")
            candidate = candidate.parent
        if candidate.is_dir():
            if (candidate / "train.csv").exists():
                return candidate
            raise FileNotFoundError(f"Directory {candidate} does not contain train.csv.")
        raise FileNotFoundError(f"Dataset path {candidate} not found.")

    default_dir = (Path(__file__).parent / "data" / "amazonproduct").resolve()
    if (default_dir / "train.csv").exists():
        return default_dir
    raise FileNotFoundError(
        "Could not locate the amazonproduct dataset. "
        "Provide --data_path pointing to a folder that contains train.csv/val.csv/test.csv."
    )


def _make_abs_path(path_value: Union[str, Path], root: Path) -> str:
    """Convert a relative path (from CSV) to an absolute path under *root*."""
    path_str = str(path_value).strip()
    if not path_str:
        return ""
    path_obj = Path(path_str)
    if not path_obj.is_absolute():
        candidates = [
            (root / path_obj).resolve(),
            (root.parent / path_obj).resolve(),
            (Path(__file__).parent / path_obj).resolve(),
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        # Fall back to root join even if it does not exist yet
        path_obj = (root / path_obj).resolve()
    return str(path_obj)


def _prepare_split(df: pd.DataFrame, *, data_root: Optional[Path]) -> pd.DataFrame:
    """Clean a dataframe split and ensure file paths are absolute."""
    result = df.copy()
    if data_root is not None and "file_path" in result.columns:
        result["file_path"] = result["file_path"].apply(lambda p: _make_abs_path(p, data_root))
    result["reviewerID"] = result["reviewerID"].astype(str)
    result["asin"] = result["asin"].astype(str)
    if "description" in result.columns:
        result["description"] = result["description"].fillna("")
    else:
        result["description"] = ""
    filtered = filter_valid_rows(result, check_images=_any_image_exists(result["file_path"]))
    if filtered.empty and not result.empty:
        print("Warning: no rows passed image validation; keeping rows without checking image files.")
        filtered = filter_valid_rows(result, check_images=False)
    return filtered


def load_amazonproduct_data(
    data_source: Optional[Union[str, Path, pd.DataFrame]] = None,
    *,
    batch_size: int = 16,
    num_workers: int = 0,
):
    """Create dataloaders for the Amazon product dataset."""

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    if isinstance(data_source, pd.DataFrame):
        df = data_source.copy()
        df["reviewerID"] = df["reviewerID"].astype(str)
        df["asin"] = df["asin"].astype(str)
        if "description" in df.columns:
            df["description"] = df["description"].fillna("")
        else:
            df["description"] = ""

        filtered_df = filter_valid_rows(df, check_images=_any_image_exists(df["file_path"]))
        if filtered_df.empty and not df.empty:
            print("Warning: no rows passed image validation; keeping rows without checking image files.")
            filtered_df = filter_valid_rows(df, check_images=False)
        df = filtered_df
        if df.empty:
            raise ValueError("Dataset is empty after filtering rows.")

        train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, shuffle=True)
        val_df, test_df = train_test_split(temp_df, test_size=2 / 3, random_state=42, shuffle=True)
    else:
        data_dir = _resolve_dataset_dir(data_source)

        def _load_split(name: str) -> pd.DataFrame:
            csv_path = data_dir / f"{name}.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"Expected split file {csv_path} is missing.")
            return pd.read_csv(csv_path)

        raw_train = _load_split("train")
        raw_val = _load_split("val")
        raw_test = _load_split("test")

        train_df = _prepare_split(raw_train, data_root=data_dir)
        val_df = _prepare_split(raw_val, data_root=data_dir)
        test_df = _prepare_split(raw_test, data_root=data_dir)
        df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        if df.empty:
            raise ValueError(
                "Dataset splits contain no valid rows after preprocessing. "
                "Ensure image files are available or disable image validation."
            )

    users = {u: i for i, u in enumerate(df["reviewerID"].astype(str).unique())}
    items = {a: i for i, a in enumerate(df["asin"].astype(str).unique())}

    full_pivot = pd.pivot_table(
        df,
        values="overall",
        index="reviewerID",
        columns="asin",
        aggfunc="mean",
        fill_value=0,
    ).astype(float)

    train_ds = AmazonReviewDataset(train_df, users, items, tokenizer, history_dim=full_pivot, max_len=128)
    val_ds = AmazonReviewDataset(val_df, users, items, tokenizer, history_dim=full_pivot, max_len=128)
    test_ds = AmazonReviewDataset(test_df, users, items, tokenizer, history_dim=full_pivot, max_len=128)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                     collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)

    return train_dl, val_dl, test_dl, users, items, full_pivot


def load_data(
    data_path: Optional[Union[str, Path, pd.DataFrame]] = None,
    *,
    data_type: str = "amazonproduct",
    batch_size: int = 16,
    num_workers: int = 0,
):
    if data_type == "amazonproduct":
        return load_amazonproduct_data(data_path, batch_size=batch_size, num_workers=num_workers)
    raise ValueError(f"Unsupported data_type: {data_type}")