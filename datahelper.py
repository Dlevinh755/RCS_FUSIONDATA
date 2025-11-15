import pandas as pd
from pathlib import Path
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import gc

IMG_SIZE = 224

# Memory-efficient image transform
img_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class AmazonReviewDataset(Dataset):
    def __init__(self, df: pd.DataFrame, user2idx, item2idx, tokenizer, 
                 history_dim, max_len=128, use_history=False, enable_cache=False):
        self.df = df.reset_index(drop=True)
        self.n_items = len(item2idx)
        self.user2idx = user2idx
        self.item2idx = item2idx
        self.tok = tokenizer
        self.max_len = max_len
        self.enable_cache = enable_cache
        self._reported_missing_images = set()
        
        if enable_cache:
            print("⚠️  Caching enabled - may use significant RAM")
            self._image_cache = {}
            self._text_cache = {}
            self._preload_data()
        else:
            print("⚡ Memory-efficient mode - processing on-the-fly")

    def _load_image_tensor(self, file_path: str) -> torch.Tensor:
        path = str(file_path).strip()
        if not Path(path).exists():
            if path not in self._reported_missing_images:
                self._reported_missing_images.add(path)
            return torch.zeros(3, IMG_SIZE, IMG_SIZE)
        
        try:
            with Image.open(path) as img:
                img_rgb = img.convert('RGB')
                return img_tf(img_rgb)
        except Exception:
            if path not in self._reported_missing_images:
                self._reported_missing_images.add(path)
            return torch.zeros(3, IMG_SIZE, IMG_SIZE)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        user_id = str(row['reviewerID'])
        current_asin = str(row['asin'])
        
        # Get indices
        uid = self.user2idx[user_id]
        iid = self.item2idx[current_asin]
        
        # Process text on-the-fly for memory efficiency
        text = str(row.get('description', ""))
        if not text or text == 'nan':
            text = ""
        
        # Tokenize
        tokens = self.tok(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        
        # Load image
        img_tensor = self._load_image_tensor(row['file_path'])
        
        # Process price
        price_value = row.get('price', 0.0)
        try:
            price = float(price_value)
        except (TypeError, ValueError):
            price = 0.0
        
        return {
            'user_idx': torch.tensor(uid, dtype=torch.long),
            'item_idx': torch.tensor(iid, dtype=torch.long),
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'image': img_tensor,
            'rating': torch.tensor(float(row['overall']), dtype=torch.float32),
            'price': torch.tensor(price, dtype=torch.float32),
        }

    def __len__(self):
        return len(self.df)
