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

def download_gz(url, out_path):
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                shutil.copyfileobj(r.raw, f, length=1<<20)
        print("Saved:", out_path)
    except requests.RequestException as e:
        print("Download error:", e)

def read_jsonlines_robust(path_gz, limit=None):
    rows = []
    with gzip.open(path_gz, 'rt', encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)          # JSON chuẩn
            except json.JSONDecodeError:
                try:
                    obj = ast.literal_eval(s)  # “Python literal” kiểu UCSD
                except Exception:
                    continue                  # bỏ qua dòng hỏng
            rows.append(obj)
            if limit and len(rows) >= limit:
                break
    return pd.DataFrame(rows)

def flatten_categories(cat_col):
    if not isinstance(cat_col, list):
        return []
    flat = []
    for c in cat_col:
        if isinstance(c, list):
            flat.extend(c)
        else:
            flat.append(c)
    return [x.strip() for x in flat]
    

def assign_label(cats):
    if not isinstance(cats, list):
        return None
    # flatten
    cats_flat = []
    for c in cats:
        if isinstance(c, list): cats_flat.extend(c)
        else: cats_flat.append(c)
    cats_flat_lower = [c.lower() for c in cats_flat]

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
    # fixed_targets, ratios, label_map should be passed in at call time to avoid
    # referencing globals that may not be defined at function-definition time.
    if fixed_targets is None:
        fixed_targets = {"men":1200, "women":1500, "shoes":1200, "watches":995}
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
    out = pd.concat(parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)


    print("===> Sampling mode:", mode)
    print("Available:", avail)
    print("Targets:", targets, "| Total:", sum(targets.values()))
    print("Result distribution:")
    print(out[label_col].value_counts())
    if label_map is None:
        raise ValueError("label_map must be provided")
    inv_map = {v:k for k,v in label_map.items()}
    out["cat_label"] = out[label_col].map(inv_map)
    out = out.drop(columns=[label_col])
    return out

def down_load_img(temp): 
    output_dir = 'data/amazon product/images/'
    os.makedirs(output_dir, exist_ok=True)
    # minimal feedback
    if os.path.exists(output_dir):
        print("Images dir:", output_dir)
         
    cnt = 0
    sucess = []
    for i, row in temp.iterrows():
        try:
            img_id   = row['asin']
            img_url  = row['imUrl']
            response = requests.get(img_url)
            img = Image.open(BytesIO(response.content))
            img.save(f"{output_dir}/{img_id}.jpg")
            cnt += 1
            sucess.append(1)
            if cnt%100 ==0:
                print(f"Tải {cnt} ảnh thành công")
        except:
            sucess.append(0)
            pass
    return sucess


def main(args):
    mode = args.mode
    data_dir = "data/amazon product"
    os.makedirs(data_dir, exist_ok=True)
    urls = [args.reviews_link, args.meta_link]
    outs = [os.path.join(data_dir, "reviews.json.gz"), os.path.join(data_dir, "metadata.json.gz")]

    for u, o in zip(urls, outs):
        download_gz(u, o)


    meta = read_jsonlines_robust(os.path.join(data_dir, "metadata.json.gz"))
    reviews = read_jsonlines_robust(os.path.join(data_dir, "reviews.json.gz"))

    # Remove old images dir if exists (safer cross-platform than shell rm)
    images_dir = os.path.join(data_dir, "images")
    shutil.rmtree(images_dir, ignore_errors=True)
    print("Removed old images directory (if existed):", images_dir)
 
    meta = meta[["asin","title", "price", "categories","description","imUrl"]].dropna()
    meta["categories"] = meta["categories"].apply(flatten_categories)
    
    temp = meta.copy()

    temp["cat_label"] = temp["categories"].apply(assign_label)
    print(temp["cat_label"].value_counts())

    counter = Counter()
    temp["categories"].apply(lambda lst: counter.update(lst))


    top_tags = counter.most_common(20)
    top_df = pd.DataFrame(top_tags, columns=["Tag", "Count"])
    print("\nTop 20 tags phổ biến nhất:")
    print(top_df)

    label_map = {
        "Men Clothing": "men",
        "Women Clothing": "women",
        "Shoes": "shoes",
        "Watches": "watches",
    }
    temp["_label"] = temp["cat_label"].map(label_map)

    target_ratio = {
        "men":    1200/4895, 
        "women":  1500/4895,
        "shoes":  1200/4895,
        "watches": 995/4895,
    }
 
    fixed_targets_main = {"men":1200, "women":1500, "shoes":1200, "watches":995}
    df_ratio = sample_like_dcares(temp, mode="ratio", seed=42, ratios=target_ratio, label_map=label_map)
    df_fixed = sample_like_dcares(temp, mode="fixed", seed=42, fixed_targets=fixed_targets_main, label_map=label_map)
 

    test = df_ratio
    sucess = down_load_img(test)
    test["sucess"] = sucess
    meta_out = test[test["sucess"] == 1]
    meta_out[["asin","title", "price","cat_label", "categories","description","imUrl"]].to_csv("data/amazon product/meta.csv", index = False)

    df_reviews = reviews
    assert "asin" in meta_out.columns, "df_meta phải có cột 'asin'"
    assert "asin" in df_reviews.columns, "df_reviews phải có cột 'asin'"

    # Lọc bỏ các hàng không có asin (NaN)
    df_reviews = df_reviews[df_reviews["asin"].notna()]
    df_reviews.drop_duplicates(subset=["reviewerID", "asin"], inplace = True)


    meta_asin_set = set(meta_out["asin"].unique())
    df_reviews_filtered = df_reviews[df_reviews["asin"].isin(meta_asin_set)].copy()


    df_reviews_filtered = df_reviews_filtered.drop_duplicates(subset=["reviewerID", "asin"])
    df_reviews_filtered = df_reviews_filtered[df_reviews_filtered["overall"].between(1,5)]
    df_reviews_filtered.to_csv("data/amazon product/reviews.csv", index = False)


    df = df_reviews_filtered.merge(meta_out, on="asin")
    df["file_path"] = "data/amazon product/images/"+ df["asin"] +".jpg"
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, shuffle=True)
    val_df, test_df = train_test_split(temp_df, test_size=(2/3), random_state=42, shuffle=True)

    train_df.to_csv("data/amazon product/train.csv", index=False)
    val_df.to_csv("data/amazon product/val.csv", index=False)
    test_df.to_csv("data/amazon product/test.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="ratio", choices=["ratio", "fixed"], help="Sampling mode: 'ratio' or 'fixed'")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--meta_link", type=str, default="https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Clothing_Shoes_and_Jewelry.json.gz", help="Link to metadata gz file")
    parser.add_argument("--reviews_link", type=str, default="https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Clothing_Shoes_and_Jewelry.json.gz", help="Link to reviews gz file")
    args = parser.parse_args()
    main(args)