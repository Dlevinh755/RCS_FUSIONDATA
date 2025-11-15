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
    def __init__(self, df: pd.DataFrame, user2idx, item2idx, tokenizer, history_dim, max_len=128, use_history=False):
        self.df = df.reset_index(drop=True)
        self.pivot_df = history_dim
        self.n_items = len(item2idx)
        self.user2idx = user2idx
        self.item2idx = item2idx
        self.tok = tokenizer
        self.max_len = max_len
        self.use_history = use_history
        self._reported_missing_images = set()
        
        self._image_cache = {}
        self._text_cache = {}
        self._history_cache = {} if use_history else None 
        
        print("Pre-caching images and text...")
        self._preload_data()
    
    def _preload_data(self):
        from tqdm import tqdm
        
        # Cache unique images
        unique_paths = self.df['file_path'].unique()
        for path in tqdm(unique_paths, desc="Caching images"):
            self._image_cache[path] = self._load_image_tensor(path)
        
        # Cache unique texts  
        unique_texts = self.df['description'].fillna("").astype(str).unique()
        for text in tqdm(unique_texts, desc="Caching texts"):
            if text not in self._text_cache:
                enc = self.tok(
                    text,
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_len,
                    return_tensors='pt'
                )
                self._text_cache[text] = {
                    'input_ids': enc['input_ids'].squeeze(0),
                    'attention_mask': enc['attention_mask'].squeeze(0)
                }
        
        # 👈 Only pre-compute historical ratings if needed
        if self.use_history and self._history_cache is not None:
            unique_users = self.df['reviewerID'].astype(str).unique()
            for user_id in tqdm(unique_users, desc="Caching user histories"):
                self._history_cache[user_id] = self._compute_history_tensor(user_id)
            print("Historical ratings cached")
        else:
            print("Skipping user history caching (not needed for this model)")
    
    def _compute_history_tensor(self, user_id: str) -> torch.Tensor:
        """Pre-compute historical ratings tensor for a user"""
        hist_tensor = torch.zeros(self.n_items, dtype=torch.float32)
        
        if user_id in self.pivot_df.index:
            historical_ratings = self.pivot_df.loc[user_id].fillna(0.0)
            for asin, rating in historical_ratings.items():
                if asin in self.item2idx:
                    item_idx = self.item2idx[asin]
                    hist_tensor[item_idx] = rating
        
        return hist_tensor
    
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
        
        # Use cached data
        text = str(row.get('description', ""))
        text_data = self._text_cache.get(text, self._text_cache[""])
        
        img_tensor = self._image_cache.get(row['file_path'], torch.zeros(3, IMG_SIZE, IMG_SIZE))
        
        # 👈 Only compute history if needed
        if self.use_history and self._history_cache is not None:
            hist_tensor = self._history_cache[user_id].clone()
            if current_asin in self.item2idx:
                hist_tensor[self.item2idx[current_asin]] = 0.0
        else:
            hist_tensor = torch.zeros(self.n_items, dtype=torch.float32)  # Empty history
        
        price_value = row.get('price', 0.0)
        try:
            price = float(price_value)
        except (TypeError, ValueError):
            price = 0.0
        
        return {
            'user_idx': torch.tensor(uid, dtype=torch.long),
            'item_idx': torch.tensor(iid, dtype=torch.long),
            'input_ids': text_data['input_ids'],
            'attention_mask': text_data['attention_mask'],
            'image': img_tensor,
            'rating': torch.tensor(float(row['overall']), dtype=torch.float32),
            'historical_ratings': hist_tensor,
            'price': torch.tensor(price, dtype=torch.float32),
        }

    def get_user_history_tensor(self, user_id: str, current_asin: str = None) -> torch.Tensor:
        """Get historical ratings tensor for a specific user"""
        if not self.use_history:
            return torch.zeros(self.n_items, dtype=torch.float32)
            
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
