import pandas as pd
import numpy as np
from pathlib import Path
import torch
from torchvision import  transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

IMG_SIZE = 224
img_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
    transforms.ConvertImageDtype(torch.float),
    transforms.Lambda(lambda x: x if x.ndim==3 else x.expand(3, *x.shape[1:])),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])


class AmazonReviewDataset(Dataset):
    def __init__(self, df: pd.DataFrame, user2idx, item2idx, tokenizer, max_len=128):
        self.df = df.reset_index(drop=True)
        self.user2idx, self.item2idx = user2idx, item2idx
        self.tok = tokenizer
        self.max_len = max_len

    def _load_image_tensor(self, file_path: str) -> torch.Tensor:
        path = str(file_path).strip()
        if not Path(path).exists():
            return torch.zeros(3, IMG_SIZE, IMG_SIZE)
        with Image.open(path).convert('RGB') as im:
            t = img_tf(torch.tensor(np.array(im)).permute(2,0,1) / 255.0)
            return t

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        uid = self.user2idx[row['reviewerID']]
        iid = self.item2idx[row['asin']]
        y = float(row['overall'])
        text = str(row['description'])
        # Tokenize RoBERTa
        enc = self.tok(
            text, padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt'
        )
        # Image
        img_tensor = self._load_image_tensor(row['file_path'])  # (3,224,224)
        sample = {
            'user_idx': torch.tensor(uid, dtype=torch.long),
            'item_idx': torch.tensor(iid, dtype=torch.long),
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'image': img_tensor,
            'rating': torch.tensor(y, dtype=torch.float32),
        }
        return sample

    def __len__(self):
        return len(self.df)
    

def filter_valid_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[df['description'].astype(str).str.strip().ne('')]
    df = df[df['file_path'].astype(str).str.strip().ne('')]
    # giữ các file ảnh tồn tại
    def image_exists(fp):
        return Path(str(fp).strip()).exists()
    df = df[df['file_path'].apply(image_exists)]
    # rating phải là số
    df = df[pd.to_numeric(df['overall'], errors='coerce').notnull()]
    return df

