import requests, shutil, argparse
import pandas as pd
import gzip, json, ast
import os
from PIL import Image
from io import BytesIO
from collections import Counter
import math
import numpy as np
from sklearn.model_selection import train_test_split
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_gz(url, out_path):
    try:
        logger.info(f"Downloading from {url}...")
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                shutil.copyfileobj(r.raw, f, length=1<<20)
        logger.info(f"Saved: {out_path}")
        return True
    except requests.RequestException as e:
        logger.error(f"Download error: {e}")
        return False

def read_jsonlines_robust(path_gz, limit=None):
    """Read JSON lines from gzipped file with robust error handling."""
    if not os.path.exists(path_gz):
        logger.error(f"File not found: {path_gz}")
        return pd.DataFrame()
    
    rows = []
    skipped = 0
    with gzip.open(path_gz, 'rt', encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)          # JSON chuẩn
            except json.JSONDecodeError:
                try:
                    obj = ast.literal_eval(s)  # "Python literal" kiểu UCSD
                except Exception as e:
                    skipped += 1
                    continue                  # bỏ qua dòng hỏng
            rows.append(obj)
            if limit and len(rows) >= limit:
                break
    
    if skipped > 0:
        logger.warning(f"Skipped {skipped} malformed lines in {path_gz}")
    logger.info(f"Read {len(rows)} records from {path_gz}")
    return pd.DataFrame(rows)

def flatten_categories(cat_col):
    """Flatten nested category lists."""
    if not isinstance(cat_col, list):
        return []
    flat = []
    for c in cat_col:
        if isinstance(c, list):
            flat.extend(c)
        else:
            flat.append(c)
    return [str(x).strip() for x in flat if x]  # Convert to string and filter empty
    
def assign_label(cats):
    """Assign category label based on hierarchy."""
    if not isinstance(cats, list):
        return None
    # flatten
    cats_flat = []
    for c in cats:
        if isinstance(c, list): 
            cats_flat.extend(c)
        else: 
            cats_flat.append(c)
    cats_flat_lower = [str(c).lower() for c in cats_flat]

    # ưu tiên cụ thể hơn trước
    if "watches" in cats_flat_lower:
        return "Watches"
    elif "shoes" in cats_flat_lower:
        return "Shoes"
    elif any("women" in c for c in cats_flat_lower):
        return "Women Clothing"
    elif any("men" in c for c in cats_flat_lower):
        return "Men Clothing"
    return None

def sample_like_dcares(
    df, label_col="_label", mode="ratio", seed=42,
    fixed_targets=None, ratios=None, label_map=None
):
    """Sample data based on ratio or fixed targets."""
    if fixed_targets is None:
        fixed_targets = {"men":1200, "women":1500, "shoes":1200, "watches":995}
    if label_map is None:
        raise ValueError("label_map must be provided")
        
    rng = np.random.RandomState(seed)
    avail = df[label_col].value_counts().to_dict()

    if mode == "ratio":
        if ratios is None:
            raise ValueError("ratios must be provided in 'ratio' mode")
        caps = []
        for k, r in ratios.items():
            if r <= 0:
                continue
            if k not in avail or avail[k] == 0:
                caps.append(0)
            else:
                caps.append(avail[k] / r)
        total_max = math.floor(min(caps)) if caps else 0
        targets = {k: int(round(r * total_max)) for k, r in ratios.items()}
        diff = total_max - sum(targets.values())
        keys = list(ratios.keys())
        i = 0
        while diff != 0 and keys:
            k = keys[i % len(keys)]
            if diff > 0 and targets[k] < avail.get(k, 0):
                targets[k] += 1; diff -= 1
            elif diff < 0 and targets[k] > 0:
                targets[k] -= 1; diff += 1
            i += 1

    elif mode == "fixed":
        # use fixed_targets keys (labels like 'men','women',...)
        targets = {k: min(int(fixed_targets.get(k, 0)), int(avail.get(k, 0))) for k in fixed_targets.keys()}
    else:
        raise ValueError("mode must be 'ratio' or 'fixed'")

    # Thực hiện sample
    parts = []
    for k, n in targets.items():
        if n <= 0:
            continue
        sub = df[df[label_col] == k]
        if len(sub) <= n:
            parts.append(sub)
        else:
            parts.append(sub.sample(n=n, random_state=seed))
    
    if not parts:
        logger.warning("No data sampled!")
        return pd.DataFrame()
        
    out = pd.concat(parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    logger.info(f"===> Sampling mode: {mode}")
    logger.info(f"Available: {avail}")
    logger.info(f"Targets: {targets} | Total: {sum(targets.values())}")
    logger.info("Result distribution:")
    logger.info(f"\n{out[label_col].value_counts()}")
    
    inv_map = {v:k for k,v in label_map.items()}
    out["cat_label"] = out[label_col].map(inv_map)
    out = out.drop(columns=[label_col])
    return out

def download_images(df, output_dir='data/amazon_product/images/'): 
    """Download images from URLs in dataframe."""
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Images directory: {output_dir}")
         
    cnt = 0
    success = []
    
    for i, row in df.iterrows():
        try:
            img_id = row['asin']
            img_url = row['imUrl']
            
            if pd.isna(img_url) or not str(img_url).startswith('http'):
                success.append(0)
                continue
                
            response = requests.get(img_url, timeout=10)
            response.raise_for_status()
            
            img = Image.open(BytesIO(response.content))
            # Validate image
            img.verify()
            # Reopen after verify
            img = Image.open(BytesIO(response.content))
            
            img_path = os.path.join(output_dir, f"{img_id}.jpg")
            img.save(img_path)
            cnt += 1
            success.append(1)
            
            if cnt % 100 == 0:
                logger.info(f"Downloaded {cnt} images successfully")
                
        except Exception as e:
            logger.debug(f"Failed to download image for {row.get('asin', 'unknown')}: {str(e)}")
            success.append(0)
    
    logger.info(f"Total images downloaded: {cnt}/{len(df)}")
    return success


def main(args):
    mode = args.mode
    data_dir = "data/amazon_product"
    os.makedirs(data_dir, exist_ok=True)
    
    # Download files
    urls = [args.reviews_link, args.meta_link]
    outs = [os.path.join(data_dir, "reviews.json.gz"), os.path.join(data_dir, "metadata.json.gz")]

    for u, o in zip(urls, outs):
        if not os.path.exists(o):
            success = download_gz(u, o)
            if not success:
                logger.error(f"Failed to download {u}")
                return
        else:
            logger.info(f"File already exists: {o}")

    # Read data
    logger.info("Reading metadata...")
    meta = read_jsonlines_robust(outs[1])
    logger.info("Reading reviews...")
    reviews = read_jsonlines_robust(outs[0])
    
    if meta.empty or reviews.empty:
        logger.error("Failed to read data files")
        return

    # Remove old images dir
    images_dir = os.path.join(data_dir, "images")
    if os.path.exists(images_dir):
        shutil.rmtree(images_dir, ignore_errors=True)
        logger.info(f"Removed old images directory: {images_dir}")
 
    # Process metadata
    required_cols = ["asin", "title", "price", "categories", "description", "imUrl"]
    missing_cols = [col for col in required_cols if col not in meta.columns]
    if missing_cols:
        logger.error(f"Missing required columns in metadata: {missing_cols}")
        return
        
    meta = meta[required_cols].dropna()
    logger.info(f"Metadata records after filtering: {len(meta)}")
    
    meta["categories"] = meta["categories"].apply(flatten_categories)
    
    # Assign labels
    meta["cat_label"] = meta["categories"].apply(assign_label)
    logger.info(f"\nCategory distribution:\n{meta['cat_label'].value_counts()}")

    # Analyze categories
    counter = Counter()
    meta["categories"].apply(lambda lst: counter.update(lst))

    top_tags = counter.most_common(20)
    top_df = pd.DataFrame(top_tags, columns=["Tag", "Count"])
    logger.info(f"\nTop 20 tags:\n{top_df}")

    # Map labels
    label_map = {
        "Men Clothing": "men",
        "Women Clothing": "women",
        "Shoes": "shoes",
        "Watches": "watches",
    }
    meta["_label"] = meta["cat_label"].map(label_map)

    # Define sampling parameters
    target_ratio = {
        "men":    1200/4895, 
        "women":  1500/4895,
        "shoes":  1200/4895,
        "watches": 995/4895,
    }
 
    fixed_targets_main = {"men":1200, "women":1500, "shoes":1200, "watches":995}
    
    # Sample data
    if mode == "ratio":
        sampled_data = sample_like_dcares(meta, mode="ratio", seed=args.seed, 
                                         ratios=target_ratio, label_map=label_map)
    elif mode == "fixed":
        sampled_data = sample_like_dcares(meta, mode="fixed", seed=args.seed, 
                                         fixed_targets=fixed_targets_main, label_map=label_map)
    else:
        sampled_data = meta
    
    if sampled_data.empty:
        logger.error("No data after sampling")
        return
    
    # Download images
    success = download_images(sampled_data)
    sampled_data["success"] = success
    meta_out = sampled_data[sampled_data["success"] == 1]
    
    logger.info(f"Successfully downloaded {len(meta_out)} images")
    
    # Save metadata
    meta_out[["asin", "title", "price", "cat_label", "categories", "description", "imUrl"]].to_csv(
        os.path.join(data_dir, "meta.csv"), index=False
    )
    logger.info(f"Saved metadata to {os.path.join(data_dir, 'meta.csv')}")

    # Process reviews
    df_reviews = reviews
    
    if "asin" not in df_reviews.columns or "asin" not in meta_out.columns:
        logger.error("Missing 'asin' column in reviews or metadata")
        return

    # Filter reviews
    df_reviews = df_reviews[df_reviews["asin"].notna()]
    df_reviews = df_reviews.drop_duplicates(subset=["reviewerID", "asin"])

    meta_asin_set = set(meta_out["asin"].unique())
    df_reviews_filtered = df_reviews[df_reviews["asin"].isin(meta_asin_set)].copy()

    if "overall" in df_reviews_filtered.columns:
        df_reviews_filtered = df_reviews_filtered[df_reviews_filtered["overall"].between(1, 5)]
    
    logger.info(f"Filtered reviews: {len(df_reviews_filtered)}")
    df_reviews_filtered.to_csv(os.path.join(data_dir, "reviews.csv"), index=False)

    # Merge and split
    df = df_reviews_filtered.merge(meta_out, on="asin")
    df["file_path"] = df["asin"].apply(lambda x: os.path.join(data_dir, "images", f"{x}.jpg"))
    
    logger.info(f"Final dataset size: {len(df)}")
    
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=args.seed, shuffle=True)
    val_df, test_df = train_test_split(temp_df, test_size=(2/3), random_state=args.seed, shuffle=True)

    train_df.to_csv(os.path.join(data_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(data_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(data_dir, "test.csv"), index=False)
    
    logger.info(f"Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    logger.info("✅ Processing completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Amazon product dataset")
    parser.add_argument("--mode", type=str, default="None", choices=["ratio", "fixed", "None"], 
                       help="Sampling mode: 'ratio' or 'fixed'")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--meta_link", type=str, 
                       default="https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Clothing_Shoes_and_Jewelry.json.gz", 
                       help="Link to metadata gz file")
    parser.add_argument("--reviews_link", type=str, 
                       default="https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Clothing_Shoes_and_Jewelry.json.gz", 
                       help="Link to reviews gz file")
    args = parser.parse_args()
    main(args)