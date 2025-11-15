import pandas as pd
from pathlib import Path
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

IMG_SIZE = 224
# Transform cho PIL Image
img_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet normalization
    transforms.Lambda(lambda x: x if x.shape[0] == 3 else x.expand(3, *x.shape[1:]))
])


class AmazonReviewDataset(Dataset):
    def __init__(self, df: pd.DataFrame, user2idx, item2idx, tokenizer, history_dim, max_len=128):
        self.df = df.reset_index(drop=True)

        # tạo user-item rating matrix
        self.pivot_df = history_dim
        
        # Store the number of items for consistent history size
        self.n_items = len(item2idx)

        self.user2idx = user2idx
        self.item2idx = item2idx
        self.tok = tokenizer
        self.max_len = max_len
        self._reported_missing_images = set()

    def _load_image_tensor(self, file_path: str) -> torch.Tensor:
        path = str(file_path).strip()
        if not Path(path).exists():
            if path not in self._reported_missing_images:
                print(f"Image not found: {path}")
                self._reported_missing_images.add(path)
            return torch.zeros(3, IMG_SIZE, IMG_SIZE)
        try:
            with Image.open(path) as im:
                im = im.convert('RGB')
                return img_tf(im)
        except Exception as e:
            if path not in self._reported_missing_images:
                print(f"Error loading image {path}: {e}")
                self._reported_missing_images.add(path)
            return torch.zeros(3, IMG_SIZE, IMG_SIZE)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        user_id = str(row['reviewerID'])
        current_asin = str(row['asin'])
        uid = self.user2idx[user_id]
        iid = self.item2idx[current_asin]

        # lấy history của đúng user này
        price_value = row.get('price', 0.0)
        try:
            price = float(price_value)
        except (TypeError, ValueError):
            price = 0.0
        if user_id in self.pivot_df.index:
            historical_ratings = self.pivot_df.loc[user_id].fillna(0.0).copy()
        else:
            historical_ratings = pd.Series(0.0, index=self.pivot_df.columns, dtype=float)
        
        # Mask current item using ASIN, not index
        if current_asin in historical_ratings.index:
            historical_ratings.loc[current_asin] = 0.0
         
        # Create a fixed-size tensor for all items
        hist_tensor = torch.zeros(self.n_items, dtype=torch.float32)
        # Map the pivot columns to the correct item indices
        for asin, rating in historical_ratings.items():
            if asin in self.item2idx:
                item_idx = self.item2idx[asin]
                hist_tensor[item_idx] = rating

        y = float(row['overall'])
        text = str(row.get('description', ""))  # hoặc 'description'

        enc = self.tok(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )

        img_tensor = self._load_image_tensor(row['file_path'])

        sample = {
            'user_idx': torch.tensor(uid, dtype=torch.long),
            'item_idx': torch.tensor(iid, dtype=torch.long),
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            
            'image': img_tensor,
            'rating': torch.tensor(y, dtype=torch.float32),
            'historical_ratings': hist_tensor,
            'price': torch.tensor(price, dtype=torch.float32),
        }
        return sample

    def __len__(self):
        return len(self.df)
    
    def get_user_history_tensor(self, user_id: str, current_asin: str = None) -> torch.Tensor:
        """Get historical ratings tensor for a specific user"""
        user_id = str(user_id)
        current_asin = str(current_asin) if current_asin is not None else None
        if user_id not in self.pivot_df.index:
            return torch.zeros(self.n_items, dtype=torch.float32)
        historical_ratings = self.pivot_df.loc[user_id].fillna(0.0).copy()

        # Mask current item if provided
        if current_asin and current_asin in historical_ratings.index:
            historical_ratings.loc[current_asin] = 0.0

        # Create fixed-size tensor
        hist_tensor = torch.zeros(self.n_items, dtype=torch.float32)
        for asin, rating in historical_ratings.items():
            if asin in self.item2idx:
                item_idx = self.item2idx[asin]
                hist_tensor[item_idx] = rating

        return hist_tensor

def filter_valid_rows(df: pd.DataFrame, *, check_images: bool = True) -> pd.DataFrame:
    df = df.copy()
    df = df[df['description'].astype(str).str.strip().ne('')]
    df = df[df['file_path'].astype(str).str.strip().ne('')]

    if check_images:
        def image_exists(fp):
            path_str = str(fp).strip()
            if not path_str:
                return False
            return Path(path_str).exists()

        df = df[df['file_path'].apply(image_exists)]

    df['overall'] = pd.to_numeric(df['overall'], errors='coerce')
    df = df[df['overall'].notnull()]
    df = df[(df['overall'] >= 1) & (df['overall'] <= 5)]

    return df.reset_index(drop=True)

