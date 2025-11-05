import requests, shutil, argparse, pandas as pd, gzip, os, numpy as np, logging
from PIL import Image
from io import BytesIO
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Auto-select fastest JSON parser
try:
    import orjson; json_loads = orjson.loads; USE_ORJSON = True
except ImportError:
    import json; json_loads = json.loads; USE_ORJSON = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info(f"Using {'orjson (5x faster)' if USE_ORJSON else 'json (pip install orjson for speedup)'}")

def download_gz(url, out_path):
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                shutil.copyfileobj(r.raw, f, length=1<<20)
        return True
    except: return False

def read_jsonlines(path_gz, chunk_size=10000):
    if not os.path.exists(path_gz): return pd.DataFrame()
    rows, skipped = [], 0
    with gzip.open(path_gz, 'rt', encoding='utf-8', errors='replace') as f:
        while True:
            chunk = [line for _ in range(chunk_size) if (line := f.readline().strip())]
            if not chunk: break
            for line in chunk:
                try: rows.append(json_loads(line))
                except: skipped += 1
    logger.info(f"Read {len(rows):,} records, skipped {skipped}")
    return pd.DataFrame(rows)

def flatten_cats(cat):
    if not isinstance(cat, list): return []
    flat = []
    for c in cat: flat.extend(c) if isinstance(c, list) else flat.append(c)
    return [str(x).strip() for x in flat if x]

def assign_label(cats):
    if not isinstance(cats, list): return None
    cats_lower = [str(c).lower() for c in flatten_cats(cats)]
    if "watches" in cats_lower: return "Watches"
    elif "shoes" in cats_lower: return "Shoes"
    elif any("women" in c for c in cats_lower): return "Women Clothing"
    elif any("men" in c for c in cats_lower): return "Men Clothing"
    return None

def get_session():
    s = requests.Session()
    s.mount("http://", HTTPAdapter(max_retries=Retry(total=2, backoff_factor=0.3), pool_connections=100, pool_maxsize=100))
    s.mount("https://", HTTPAdapter(max_retries=Retry(total=2, backoff_factor=0.3), pool_connections=100, pool_maxsize=100))
    s.headers.update({'User-Agent': 'Mozilla/5.0', 'Connection': 'keep-alive'})
    return s

def download_img(data, out_dir, session):
    idx, img_id, img_url = data
    try:
        if isinstance(img_url, (list, np.ndarray)): img_url = img_url[0] if len(img_url) > 0 else None
        if not img_url or not str(img_url).startswith('http'): return (idx, 0)
        
        img_path = os.path.join(out_dir, f"{img_id}.jpg")
        if os.path.exists(img_path): return (idx, 1)
        
        img = Image.open(BytesIO(session.get(str(img_url), timeout=5).content))
        if img.format in ['JPEG', 'PNG', 'JPG', 'WEBP']:
            (img.convert('RGB') if img.mode != 'RGB' else img).save(img_path, 'JPEG', quality=85, optimize=False)
            return (idx, 1)
    except: pass
    return (idx, 0)

def download_images(df, out_dir='data/amazon_product/images/', workers=50):
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*60}\n🖼️  DOWNLOADING {len(df):,} IMAGES\n{'='*60}")
    
    data = [(i, r['asin'], r['imUrl']) for i, r in df.iterrows()]
    sessions = {i: get_session() for i in range(workers)}
    success, cnt = {}, 0
    
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(download_img, d, out_dir, sessions[hash(d[0]) % workers]): d[0] for d in data}
        for i, future in enumerate(as_completed(futures), 1):
            idx, status = future.result()
            success[idx] = status
            if status: cnt += 1
            if i % max(1, len(df) // 20) == 0 or i == len(df):
                print(f"Progress: {i:>6,}/{len(df):,} ({i/len(df)*100:>5.1f}%) - Success: {cnt:>6,} ✓")
    
    for s in sessions.values(): s.close()
    print(f"\n✅ Complete: {cnt:,}/{len(df):,} ({cnt/len(df)*100:.1f}%)\n{'='*60}\n")
    return [success.get(i, 0) for i in df.index]

def main(args):
    print(f"\n{'='*60}\n🚀 AMAZON DATA PREP | Mode: {args.mode} | Workers: {args.workers}\n{'='*60}\n")
    
    data_dir = "data/amazon_product"
    os.makedirs(data_dir, exist_ok=True)
    
    # Download files
    print("📥 STEP 1/6: Downloading...")
    for name, url in [("reviews.json.gz", args.reviews_link), ("metadata.json.gz", args.meta_link)]:
        path = os.path.join(data_dir, name)
        (print(f"  ✓ {name} exists") if os.path.exists(path) else (download_gz(url, path) or exit()))
    
    # Read data
    print("\n📖 STEP 2/6: Reading data...")
    meta, reviews = read_jsonlines(f"{data_dir}/metadata.json.gz"), read_jsonlines(f"{data_dir}/reviews.json.gz")
    if meta.empty or reviews.empty: return print("❌ Empty data")
    print(f"  ✓ Meta: {len(meta):,} | Reviews: {len(reviews):,}")
    
    # Clean old images
    print("\n🗑️  STEP 3/6: Cleaning...")
    shutil.rmtree(f"{data_dir}/images", ignore_errors=True)
    
    # Process metadata
    print("\n🔧 STEP 4/6: Processing...")
    col_map = {}
    if 'category' in meta.columns: col_map['categories'] = 'category'
    if 'imageURLHighRes' in meta.columns: col_map['imUrl'] = 'imageURLHighRes'
    elif 'imageURL' in meta.columns: col_map['imUrl'] = 'imageURL'
    
    cols = ['asin', 'title', 'price', 'description'] + [col_map.get(k, k) for k in ['categories', 'imUrl'] if col_map.get(k, k) in meta.columns]
    meta = meta[cols].dropna().rename(columns={v: k for k, v in col_map.items()})
    print(f"  ✓ Filtered: {len(meta):,} records")
    
    # Process categories
    def safe_cat(c):
        if c is None or (isinstance(c, float) and np.isnan(c)): return []
        if isinstance(c, str): return [c.strip()] if c.strip() else []
        if isinstance(c, (list, np.ndarray)): return flatten_cats(c.tolist() if isinstance(c, np.ndarray) else c)
        return [str(c).strip()]
    
    meta["categories"] = meta["categories"].apply(safe_cat)
    meta["imUrl"] = meta["imUrl"].apply(lambda x: x[0] if isinstance(x, (list, np.ndarray)) and len(x) > 0 else x)
    meta["cat_label"] = meta["categories"].apply(assign_label)
    
    if (valid := meta["cat_label"].notna().sum()) == 0: return print("❌ No valid labels")
    print(f"  ✓ Labels: {valid:,}/{len(meta):,}")
    for cat, cnt in meta['cat_label'].value_counts().items():
        if pd.notna(cat): print(f"    - {cat}: {cnt:,}")
    
    # Sample
    print(f"\n📊 STEP 5/6: Sampling (mode: {args.mode})...")
    print(f"  ✓ Data: {len(meta):,}")
    
    # Download images
    print(f"\n🖼️  STEP 6/6: Downloading images...")
    meta["success"] = download_images(meta, workers=args.workers)
    meta_out = meta[meta["success"] == 1]
    print(f"  ✓ Saved: {len(meta_out):,}/{len(meta):,}")
    
    # Save & merge
    meta_out.to_csv(f"{data_dir}/meta.csv", index=False)
    reviews = reviews[reviews["asin"].isin(set(meta_out["asin"]))]
    reviews = reviews.drop_duplicates(subset=["reviewerID", "asin"] if "reviewerID" in reviews else ["asin"])
    if "overall" in reviews: reviews = reviews[reviews["overall"].between(1, 5)]
    
    df = reviews.merge(meta_out, on="asin")
    df["file_path"] = df["asin"].apply(lambda x: f"{data_dir}/images/{x}.jpg")
    
    if len(df) < 10: return print(f"❌ Too few records: {len(df)}")
    
    # Split & save
    train, temp = train_test_split(df, test_size=0.30, random_state=args.seed)
    val, test = train_test_split(temp, test_size=2/3, random_state=args.seed)
    
    df.to_csv(f"{data_dir}/full_data.csv", index=False)
    train.to_csv(f"{data_dir}/train.csv", index=False)
    val.to_csv(f"{data_dir}/val.csv", index=False)
    test.to_csv(f"{data_dir}/test.csv", index=False)
    
    print(f"\n{'='*60}\n📊 FINAL STATS\n{'='*60}")
    print(f"Train: {len(train):>6,} ({len(train)/len(df)*100:>5.1f}%)")
    print(f"Val:   {len(val):>6,} ({len(val)/len(df)*100:>5.1f}%)")
    print(f"Test:  {len(test):>6,} ({len(test)/len(df)*100:>5.1f}%)")
    print(f"{'-'*60}\nTotal: {len(df):>6,} samples\n{'='*60}\n✅ COMPLETE!\n{'='*60}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="None", choices=["ratio", "fixed", "None"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=50)
    parser.add_argument("--meta_link", default="https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Clothing_Shoes_and_Jewelry.json.gz")
    parser.add_argument("--reviews_link", default="https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Clothing_Shoes_and_Jewelry.json.gz")
    main(parser.parse_args())