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
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import ujson  # Faster JSON parser (install: pip install ujson)
from multiprocessing import Pool, cpu_count
import multiprocessing as mp
import orjson  # Nhanh hơn cả ujson (install: pip install orjson)

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

def parse_json_line(line):
    """Parse a single JSON line. Returns dict or None."""
    s = line.strip()
    if not s:
        return None
    try:
        return ujson.loads(s)  # Faster than json.loads
    except (ValueError, ujson.JSONDecodeError):
        try:
            return ast.literal_eval(s)
        except Exception:
            return None

def parse_json_line_orjson(line):
    """Parse using orjson (fastest)."""
    s = line.strip()
    if not s:
        return None
    try:
        return orjson.loads(s)  # Fastest option
    except (ValueError, orjson.JSONDecodeError):
        try:
            return ast.literal_eval(s)
        except Exception:
            return None

def read_jsonlines_robust_fast(path_gz, limit=None, chunk_size=10000):
    """Read JSON lines with optimized parsing (using ujson + chunking)."""
    if not os.path.exists(path_gz):
        logger.error(f"File not found: {path_gz}")
        return pd.DataFrame()
    
    rows = []
    skipped = 0
    
    with gzip.open(path_gz, 'rt', encoding='utf-8', errors='replace') as f:
        while True:
            # Read in chunks for better memory efficiency
            chunk_lines = []
            for _ in range(chunk_size):
                line = f.readline()
                if not line:
                    break
                chunk_lines.append(line)
            
            if not chunk_lines:
                break
            
            # Parse chunk
            for line in chunk_lines:
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = ujson.loads(s)  # 2-3x faster than json.loads
                except (ValueError, ujson.JSONDecodeError):
                    try:
                        obj = ast.literal_eval(s)
                    except Exception:
                        skipped += 1
                        continue
                rows.append(obj)
                
                if limit and len(rows) >= limit:
                    break
            
            if limit and len(rows) >= limit:
                break
    
    if skipped > 0:
        logger.warning(f"Skipped {skipped} malformed lines in {path_gz}")
    logger.info(f"Read {len(rows)} records from {path_gz}")
    return pd.DataFrame(rows)

def read_jsonlines_robust_parallel(path_gz, limit=None, num_workers=None):
    """Read JSON lines with parallel processing (best for very large files)."""
    if not os.path.exists(path_gz):
        logger.error(f"File not found: {path_gz}")
        return pd.DataFrame()
    
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    
    logger.info(f"Reading {path_gz} with {num_workers} workers...")
    
    # Read all lines first (fast I/O)
    with gzip.open(path_gz, 'rt', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
        if limit:
            lines = lines[:limit * 2]  # Read a bit extra to account for bad lines
    
    # Parse in parallel
    with Pool(num_workers) as pool:
        parsed = pool.map(parse_json_line, lines, chunksize=1000)
    
    # Filter out None values
    rows = [obj for obj in parsed if obj is not None]
    
    if limit and len(rows) > limit:
        rows = rows[:limit]
    
    skipped = len(lines) - len(rows)
    if skipped > 0:
        logger.warning(f"Skipped {skipped} malformed lines in {path_gz}")
    logger.info(f"Read {len(rows)} records from {path_gz}")
    return pd.DataFrame(rows)

# Original function renamed as backup
def read_jsonlines_robust_original(path_gz, limit=None):
    """Read JSON lines from gzipped file with robust error handling (original version)."""
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

# Use the fast version by default
read_jsonlines_robust = read_jsonlines_robust_fast

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

def download_single_image(row, output_dir):
    """Download a single image. Returns (index, success_status)."""
    try:
        img_id = row['asin']
        img_url = row['imUrl']
        
        # Handle numpy array or list - extract first URL
        if isinstance(img_url, (list, np.ndarray)):
            if len(img_url) == 0:
                return (row.name, 0)
            img_url = img_url[0] if isinstance(img_url, np.ndarray) else img_url[0]
        
        # Check if valid URL
        if img_url is None or img_url == '' or not str(img_url).startswith('http'):
            return (row.name, 0)
            
        response = requests.get(str(img_url), timeout=10)
        response.raise_for_status()
        
        img = Image.open(BytesIO(response.content))
        img.verify()
        img = Image.open(BytesIO(response.content))
        
        img_path = os.path.join(output_dir, f"{img_id}.jpg")
        img.save(img_path)
        
        return (row.name, 1)
                
    except Exception as e:
        return (row.name, 0)


def download_images(df, output_dir='data/amazon_product/images/', max_workers=10): 
    """Download images from URLs in dataframe using parallel processing."""
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Images directory: {output_dir}")
    logger.info(f"Starting parallel download with {max_workers} workers...")
    
    print(f"\n{'='*60}")
    print(f"🖼️  DOWNLOADING {len(df)} IMAGES...")
    print(f"{'='*60}")
    
    success_dict = {}
    cnt = 0
    total = len(df)
    
    # Use ThreadPoolExecutor for I/O bound tasks
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        download_func = partial(download_single_image, output_dir=output_dir)
        futures = {executor.submit(download_func, row): idx 
                  for idx, row in df.iterrows()}
        
        # Process completed tasks
        for future in as_completed(futures):
            idx, status = future.result()
            success_dict[idx] = status
            
            if status == 1:
                cnt += 1
                if cnt % 100 == 0:
                    progress = cnt / total * 100
                    print(f"Progress: {cnt}/{total} ({progress:.1f}%) ✓")
                    logger.info(f"Downloaded {cnt}/{total} images successfully")
    
    # Convert dict back to list in original order
    success = [success_dict.get(idx, 0) for idx in df.index]
    
    success_rate = cnt / total * 100
    print(f"\n✅ Download complete: {cnt}/{total} images ({success_rate:.1f}%)")
    print(f"{'='*60}\n")
    
    logger.info(f"Total images downloaded: {cnt}/{total}")
    return success


def main(args):
    print("\n" + "="*60)
    print("🚀 AMAZON PRODUCT DATA PREPARATION")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"JSON Parser: {args.json_parser}")
    print(f"Max Workers: {args.max_workers}")
    print(f"Random Seed: {args.seed}")
    print("="*60 + "\n")
    
    mode = args.mode
    data_dir = "data/amazon_product"
    os.makedirs(data_dir, exist_ok=True)
    
    # Select parser based on argument
    global read_jsonlines_robust
    if args.json_parser == "parallel":
        read_jsonlines_robust = read_jsonlines_robust_parallel
    elif args.json_parser == "original":
        read_jsonlines_robust = read_jsonlines_robust_original
    else:  # fast
        read_jsonlines_robust = read_jsonlines_robust_fast
    
    # Download files
    print("📥 STEP 1/7: Downloading data files...")
    urls = [args.reviews_link, args.meta_link]
    outs = [os.path.join(data_dir, "reviews.json.gz"), os.path.join(data_dir, "metadata.json.gz")]

    for u, o in zip(urls, outs):
        if not os.path.exists(o):
            success = download_gz(u, o)
            if not success:
                logger.error(f"Failed to download {u}")
                return
        else:
            print(f"  ✓ File exists: {os.path.basename(o)}")
            logger.info(f"File already exists: {o}")

    # Read data
    print("\n📖 STEP 2/7: Reading JSON data files...")
    try:
        logger.info("Reading metadata...")
        meta = read_jsonlines_robust(outs[1])
        print(f"  ✓ Metadata: {len(meta):,} records")
        
        logger.info("Reading reviews...")
        reviews = read_jsonlines_robust(outs[0])
        print(f"  ✓ Reviews: {len(reviews):,} records")
    except Exception as e:
        print(f"  ❌ Error reading data: {e}")
        logger.error(f"Error reading data: {e}", exc_info=True)
        return
    
    if meta.empty or reviews.empty:
        print("  ❌ Empty dataframes after reading")
        logger.error("Failed to read data files")
        return

    # Remove old images dir
    print("\n🗑️  STEP 3/7: Cleaning old data...")
    images_dir = os.path.join(data_dir, "images")
    if os.path.exists(images_dir):
        shutil.rmtree(images_dir, ignore_errors=True)
        print(f"  ✓ Removed old images directory")
        logger.info(f"Removed old images directory: {images_dir}")
 
    # Process metadata
    print("\n🔧 STEP 4/7: Processing metadata...")
    try:
        # Check available columns
        print(f"  Available columns: {list(meta.columns)}")
        
        # Flexible column mapping - handle different dataset formats
        column_mapping = {}
        
        # Map category/categories
        if 'categories' in meta.columns:
            column_mapping['categories'] = 'categories'
        elif 'category' in meta.columns:
            column_mapping['categories'] = 'category'
            print(f"  ℹ️  Using 'category' instead of 'categories'")
        
        # Map image URL
        if 'imUrl' in meta.columns:
            column_mapping['imUrl'] = 'imUrl'
        elif 'imageURL' in meta.columns:
            column_mapping['imUrl'] = 'imageURL'
            print(f"  ℹ️  Using 'imageURL' instead of 'imUrl'")
        elif 'imageURLHighRes' in meta.columns:
            column_mapping['imUrl'] = 'imageURLHighRes'
            print(f"  ℹ️  Using 'imageURLHighRes' for images")
        
        # Required columns with flexible names
        base_required = ['asin', 'title', 'price', 'description']
        required_cols = base_required.copy()
        
        # Add mapped columns
        for target, source in column_mapping.items():
            if source in meta.columns:
                required_cols.append(source)
        
        missing_cols = [col for col in base_required if col not in meta.columns]
        
        if missing_cols:
            print(f"  ❌ Missing essential columns: {missing_cols}")
            logger.error(f"Missing essential columns in metadata: {missing_cols}")
            return
        
        print(f"  ✓ Using columns: {required_cols}")
        
        # Filter data
        print(f"  Filtering data...")
        initial_count = len(meta)
        meta = meta[required_cols].dropna()
        filtered_count = len(meta)
        print(f"  ✓ Filtered: {initial_count:,} → {filtered_count:,} records (removed {initial_count - filtered_count:,})")
        logger.info(f"Metadata records after filtering: {len(meta)}")
        
        if len(meta) == 0:
            print(f"  ❌ No records left after filtering!")
            return
        
        # Rename columns to standard names
        rename_dict = {v: k for k, v in column_mapping.items()}
        if rename_dict:
            meta = meta.rename(columns=rename_dict)
            print(f"  ✓ Renamed columns: {rename_dict}")
        
        # Check if we have categories column now
        if 'categories' not in meta.columns:
            print(f"  ❌ No category information available")
            print(f"  Available columns after rename: {list(meta.columns)}")
            return
        
        # Flatten categories
        print(f"  Processing categories...")
        # Handle both list and string categories
        def safe_flatten_categories(cat_col):
            # Handle None/NaN
            if cat_col is None:
                return []
            
            # Handle pandas NA/NaN - check type first
            try:
                if isinstance(cat_col, float) and np.isnan(cat_col):
                    return []
            except (TypeError, ValueError):
                pass
            
            # Handle numpy array
            if isinstance(cat_col, np.ndarray):
                cat_col = cat_col.tolist()
            
            # Handle string
            if isinstance(cat_col, str):
                if not cat_col or cat_col.strip() == '':
                    return []
                return [cat_col.strip()]
            
            # Handle list
            if isinstance(cat_col, list):
                return flatten_categories(cat_col)
            
            # Fallback - convert to string
            return [str(cat_col).strip()]
        
        meta["categories"] = meta["categories"].apply(safe_flatten_categories)
        print(f"  ✓ Categories processed")
        
        # Process image URLs - extract first URL if array
        print(f"  Processing image URLs...")
        def safe_extract_image_url(img_url):
            if img_url is None:
                return None
            if isinstance(img_url, (list, np.ndarray)):
                if len(img_url) == 0:
                    return None
                return img_url[0]
            return img_url
        
        meta["imUrl"] = meta["imUrl"].apply(safe_extract_image_url)
        print(f"  ✓ Image URLs processed")
        
        # Assign labels
        print(f"  Assigning labels...")
        meta["cat_label"] = meta["categories"].apply(assign_label)
        
        # Check if we have valid labels
        valid_labels = meta["cat_label"].notna().sum()
        print(f"  ✓ Labels assigned: {valid_labels:,}/{len(meta):,} records")
        
        if valid_labels == 0:
            print(f"  ❌ No valid category labels found!")
            print(f"  Sample categories: {meta['categories'].head(5).tolist()}")
            # Print unique categories to help debug
            all_cats = []
            for cats in meta['categories'].head(20):
                all_cats.extend(cats)
            unique_cats = list(set(all_cats))[:20]
            print(f"  Sample unique category values: {unique_cats}")
            return
        
        print(f"\n  Category distribution:")
        cat_dist = meta['cat_label'].value_counts()
        for cat, count in cat_dist.items():
            if pd.notna(cat):
                print(f"    - {cat}: {count:,}")
        logger.info(f"\nCategory distribution:\n{meta['cat_label'].value_counts()}")
        
    except Exception as e:
        print(f"  ❌ Error in STEP 4: {e}")
        logger.error(f"Error processing metadata: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        return

    # Analyze categories
    try:
        counter = Counter()
        meta["categories"].apply(lambda lst: counter.update(lst))

        top_tags = counter.most_common(20)
        top_df = pd.DataFrame(top_tags, columns=["Tag", "Count"])
        logger.info(f"\nTop 20 tags:\n{top_df}")
    except Exception as e:
        print(f"  ⚠️  Warning: Could not analyze categories: {e}")

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
    print(f"\n📊 STEP 5/7: Sampling data (mode: {mode})...")
    try:
        if mode == "ratio":
            sampled_data = sample_like_dcares(meta, mode="ratio", seed=args.seed, 
                                             ratios=target_ratio, label_map=label_map)
        elif mode == "fixed":
            sampled_data = sample_like_dcares(meta, mode="fixed", seed=args.seed, 
                                             fixed_targets=fixed_targets_main, label_map=label_map)
        else:
            print(f"  ⚠️  No sampling applied, using all data")
            sampled_data = meta
        
        if sampled_data.empty:
            print(f"  ❌ No data after sampling")
            logger.error("No data after sampling")
            return
        
        print(f"  ✓ Sampled data: {len(sampled_data):,} records")
        
    except Exception as e:
        print(f"  ❌ Error in sampling: {e}")
        logger.error(f"Error in sampling: {e}", exc_info=True)
        return
    
    # Download images
    print(f"\n🖼️  STEP 6/7: Downloading images...")
    try:
        success = download_images(sampled_data, max_workers=args.max_workers)
        sampled_data["success"] = success
        meta_out = sampled_data[sampled_data["success"] == 1]
        
        success_rate = len(meta_out) / len(sampled_data) * 100
        print(f"  ✓ Images saved: {len(meta_out):,}/{len(sampled_data):,} ({success_rate:.1f}%)")
        logger.info(f"Successfully downloaded {len(meta_out)} images")
    except Exception as e:
        print(f"  ❌ Error downloading images: {e}")
        logger.error(f"Error downloading images: {e}", exc_info=True)
        return
    
    # Save metadata
    try:
        # Build save columns list dynamically
        save_cols = ['asin']
        for col in ['title', 'price', 'cat_label', 'categories', 'description', 'imUrl']:
            if col in meta_out.columns:
                save_cols.append(col)
        
        meta_out[save_cols].to_csv(
            os.path.join(data_dir, "meta.csv"), index=False
        )
        print(f"  ✓ Metadata saved: meta.csv (columns: {save_cols})")
        logger.info(f"Saved metadata to {os.path.join(data_dir, 'meta.csv')}")
    except Exception as e:
        print(f"  ❌ Error saving metadata: {e}")
        logger.error(f"Error saving metadata: {e}", exc_info=True)

    # Process reviews
    print(f"\n📝 STEP 7/7: Processing reviews and creating splits...")
    try:
        df_reviews = reviews
        
        if "asin" not in df_reviews.columns or "asin" not in meta_out.columns:
            print(f"  ❌ Missing 'asin' column")
            logger.error("Missing 'asin' column in reviews or metadata")
            return

        # Filter reviews
        initial_reviews = len(df_reviews)
        df_reviews = df_reviews[df_reviews["asin"].notna()]
        
        # Check if reviewerID exists
        if "reviewerID" in df_reviews.columns:
            df_reviews = df_reviews.drop_duplicates(subset=["reviewerID", "asin"])
        else:
            print(f"  ⚠️  'reviewerID' not found, skipping deduplication")
            df_reviews = df_reviews.drop_duplicates(subset=["asin"])

        meta_asin_set = set(meta_out["asin"].unique())
        df_reviews_filtered = df_reviews[df_reviews["asin"].isin(meta_asin_set)].copy()

        if "overall" in df_reviews_filtered.columns:
            df_reviews_filtered = df_reviews_filtered[df_reviews_filtered["overall"].between(1, 5)]
        
        print(f"  Initial reviews: {initial_reviews:,}")
        print(f"  After filtering: {len(df_reviews_filtered):,}")
        
        # Check if we have any reviews
        if len(df_reviews_filtered) == 0:
            print(f"  ⚠️  No reviews matched with products")
            print(f"  Sample product ASINs: {list(meta_asin_set)[:5]}")
            print(f"  Sample review ASINs: {df_reviews['asin'].head(5).tolist()}")
            
            # Save what we have so far
            meta_out.to_csv(os.path.join(data_dir, "products_only.csv"), index=False)
            print(f"  ✓ Saved products to products_only.csv (no reviews matched)")
            
            print("\n" + "="*60)
            print("⚠️  PROCESSING COMPLETED WITH WARNINGS")
            print("="*60)
            print(f"Products: {len(meta_out):,}")
            print(f"Reviews:  0 (no matches)")
            print("="*60 + "\n")
            return
        
        logger.info(f"Filtered reviews: {len(df_reviews_filtered)}")
        df_reviews_filtered.to_csv(os.path.join(data_dir, "reviews.csv"), index=False)

        # Merge and split
        df = df_reviews_filtered.merge(meta_out, on="asin")
        df["file_path"] = df["asin"].apply(lambda x: os.path.join(data_dir, "images", f"{x}.jpg"))
        
        print(f"  Merged dataset: {len(df):,} records")
        logger.info(f"Final dataset size: {len(df)}")
        
        # Check if we have enough data to split
        if len(df) < 10:
            print(f"  ⚠️  Dataset too small ({len(df)} records) - saving without splitting")
            df.to_csv(os.path.join(data_dir, "full_data.csv"), index=False)
            
            print("\n" + "="*60)
            print("⚠️  PROCESSING COMPLETED WITH WARNINGS")
            print("="*60)
            print(f"Total samples: {len(df)} (too small to split)")
            print("="*60 + "\n")
            return
        
        train_df, temp_df = train_test_split(df, test_size=0.30, random_state=args.seed, shuffle=True)
        val_df, test_df = train_test_split(temp_df, test_size=(2/3), random_state=args.seed, shuffle=True)
        
        df.to_csv(os.path.join(data_dir, "full_data.csv"), index=False)
        train_df.to_csv(os.path.join(data_dir, "train.csv"), index=False)
        val_df.to_csv(os.path.join(data_dir, "val.csv"), index=False)
        test_df.to_csv(os.path.join(data_dir, "test.csv"), index=False)
        
        # Final summary
        print("\n" + "="*60)
        print("📊 FINAL DATASET STATISTICS")
        print("="*60)
        print(f"Train set:  {len(train_df):>6,} samples ({len(train_df)/len(df)*100:>5.1f}%)")
        print(f"Val set:    {len(val_df):>6,} samples ({len(val_df)/len(df)*100:>5.1f}%)")
        print(f"Test set:   {len(test_df):>6,} samples ({len(test_df)/len(df)*100:>5.1f}%)")
        print(f"{'-'*60}")
        print(f"Total:      {len(df):>6,} samples")
        print("="*60)
        print("✅ Processing completed successfully!")
        print("="*60 + "\n")
        
        logger.info(f"Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        logger.info("✅ Processing completed successfully!")
        
    except Exception as e:
        print(f"  ❌ Error in STEP 7: {e}")
        logger.error(f"Error processing reviews: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Amazon product dataset")
    parser.add_argument("--mode", type=str, default="None", choices=["ratio", "fixed", "None"], 
                       help="Sampling mode: 'ratio' or 'fixed'")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--json-parser", type=str, default="fast", 
                       choices=["fast", "parallel", "original"],
                       help="JSON parser mode: fast (ujson+chunking), parallel (multiprocessing), or original")
    parser.add_argument("--max-workers", type=int, default=10,
                       help="Number of parallel workers for image downloading")
    parser.add_argument("--meta_link", type=str, 
                       default="https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Clothing_Shoes_and_Jewelry.json.gz", 
                       help="Link to metadata gz file")
    parser.add_argument("--reviews_link", type=str, 
                       default="https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Clothing_Shoes_and_Jewelry.json.gz", 
                       help="Link to reviews gz file")
    args = parser.parse_args()
    main(args)