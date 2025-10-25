import pandas as pd
import torch
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import numpy as np
from image_model.vgg19_model import ImageEncoder
from datahelper import img_tf
import argparse
import os
from tqdm import tqdm

IMG_SIZE = 224

def load_image_tensor(file_path: str) -> torch.Tensor:
    path = Path(file_path)
    if not path.exists():
        return None
    
    try:
        with Image.open(path).convert('RGB') as im:
            return img_tf(im)
    except Exception as e:
        print(f"Lỗi khi load ảnh {file_path}: {e}")
        return None

def filter_existing_images(df: pd.DataFrame, base_path: str = None) -> pd.DataFrame:
    """Lọc bỏ các ảnh không tồn tại"""
    print(f"\nKiểm tra {len(df)} ảnh...")
    
    def check_exists(file_path):
        if base_path:
            full_path = Path(base_path) / file_path
        else:
            full_path = Path(file_path)
        return full_path.exists()
    
    # Cập nhật đường dẫn đầy đủ nếu cần
    if base_path:
        df['file_path'] = df['file_path'].apply(lambda x: str(Path(base_path) / x))
    
    existing_mask = df['file_path'].apply(lambda x: Path(x).exists())
    df_filtered = df[existing_mask].copy().reset_index(drop=True)
    
    missing_count = len(df) - len(df_filtered)
    if missing_count > 0:
        print(f"⚠️ Loại bỏ {missing_count} ảnh không tồn tại ({missing_count/len(df)*100:.1f}%)")
        print(f"✓ Còn lại {len(df_filtered)} ảnh hợp lệ")
    else:
        print(f"✓ Tất cả {len(df)} ảnh đều tồn tại")
    
    return df_filtered

def extract_image_features(df: pd.DataFrame, device='cuda', checkpoint_interval=500) -> pd.DataFrame:
    img_enc = ImageEncoder().to(device)
    img_enc.eval()
    
    features_list = []
    checkpoint_file = "train_features_checkpoint.pkl"
    
    # Kiểm tra checkpoint
    start_idx = 0
    if os.path.exists(checkpoint_file):
        try:
            print(f"Tìm thấy checkpoint, đang tải...")
            checkpoint_df = pd.read_pickle(checkpoint_file)
            if 'img_feature' in checkpoint_df.columns and len(checkpoint_df) > 0:
                start_idx = len(checkpoint_df)
                features_list = checkpoint_df['img_feature'].tolist()
                print(f"✓ Tiếp tục từ ảnh thứ {start_idx}/{len(df)}")
        except Exception as e:
            print(f"Lỗi khi load checkpoint: {e}")
            start_idx = 0
    
    if start_idx >= len(df):
        print("Đã trích xuất xong tất cả features!")
        df['img_feature'] = features_list[:len(df)]
        return df
    
    print(f"Đang trích xuất features cho {len(df) - start_idx} ảnh còn lại...")
    
    try:
        with torch.no_grad():
            for idx in tqdm(range(start_idx, len(df)), desc="Extracting features"):
                row = df.iloc[idx]
                
                img_tensor = load_image_tensor(row['file_path'])
                
                if img_tensor is None:
                    # Tạo feature zero cho ảnh lỗi
                    features_list.append(np.zeros(4096))
                    continue
                
                img_batch = img_tensor.unsqueeze(0).to(device)
                feature = img_enc(img_batch)
                feature_np = feature.cpu().numpy().flatten()
                features_list.append(feature_np)
                
                # Lưu checkpoint định kỳ
                if (idx + 1) % checkpoint_interval == 0:
                    temp_df = df.iloc[:idx+1].copy()
                    temp_df['img_feature'] = features_list
                    temp_df.to_pickle(checkpoint_file)
                    print(f"\n💾 Đã lưu checkpoint tại {idx+1}/{len(df)}")
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Đã dừng! Đang lưu checkpoint...")
        temp_df = df.iloc[:len(features_list)].copy()
        temp_df['img_feature'] = features_list
        temp_df.to_pickle(checkpoint_file)
        print(f"✓ Đã lưu {len(features_list)} features vào checkpoint")
        raise
    
    df['img_feature'] = features_list
    print(f"✓ Hoàn thành trích xuất features!")
    
    # Xóa checkpoint khi hoàn thành
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    
    return df

def find_similar_images(query_image_path: str, df: pd.DataFrame, k=6, device='cuda'):
    print(f"\n🔍 Đang tìm {k} ảnh giống với: {query_image_path}")
    
    # Trích xuất feature của ảnh query
    img_enc = ImageEncoder().to(device)
    img_enc.eval()
    
    with torch.no_grad():
        query_tensor = load_image_tensor(query_image_path)
        if query_tensor is None:
            raise ValueError(f"Không thể load ảnh query: {query_image_path}")
        query_batch = query_tensor.unsqueeze(0).to(device)
        query_feature = img_enc(query_batch).cpu().numpy().flatten()
    
    # Lấy ASIN của ảnh query (nếu có trong DataFrame)
    query_asin = None
    query_match = df[df['file_path'] == query_image_path]
    if not query_match.empty:
        query_asin = query_match.iloc[0]['asin']
        print(f"📌 ASIN của ảnh query: {query_asin}")
    
    features_matrix = np.vstack(df['img_feature'].values)
    
    print("🔎 Đang tìm kiếm bằng KNN...")
    # Tìm nhiều hơn k ảnh để có đủ sau khi lọc
    search_k = min(k * 3, len(df))
    knn = NearestNeighbors(n_neighbors=search_k, metric='cosine')
    knn.fit(features_matrix)
    distances, indices = knn.kneighbors([query_feature])
    
    # Lọc bỏ các ảnh có cùng ASIN
    similar_df = df.iloc[indices[0]].copy()
    similar_df['distance'] = distances[0]
    similar_df['similarity_score'] = 1 - distances[0]
    
    # Loại bỏ ảnh query và các ảnh có cùng ASIN
    if query_asin:
        similar_df = similar_df[similar_df['asin'] != query_asin]
        print(f"🗑️ Đã lọc bỏ {len(df.iloc[indices[0]]) - len(similar_df)} ảnh có cùng ASIN: {query_asin}")
    
    # Lấy top k kết quả
    similar_df = similar_df.head(k)
    
    return similar_df

def plot_similar_images(query_image_path: str, similar_df: pd.DataFrame, save_path=None):
    n_results = len(similar_df)
    fig, axes = plt.subplots(2, (n_results + 1) // 2, figsize=(15, 8))
    axes = axes.flatten()
    
    query_img = Image.open(query_image_path).convert('RGB')
    axes[0].imshow(query_img)
    axes[0].set_title('Query Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    for idx, (_, row) in enumerate(similar_df.iterrows(), start=1):
        if idx >= len(axes):
            break
            
        try:
            img = Image.open(row['file_path']).convert('RGB')
            axes[idx].imshow(img)
            
            title = f"#{idx}\n"
            title += f"Similarity: {row['similarity_score']:.3f}\n"
            title += f"Rating: {row['overall']:.1f}\n"
            title += f"ASIN: {row['asin']}"
            
            axes[idx].set_title(title, fontsize=10)
            axes[idx].axis('off')
        except Exception as e:
            print(f"Lỗi khi hiển thị ảnh {row['file_path']}: {e}")
            axes[idx].axis('off')
    
    for idx in range(len(similar_df) + 1, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Đã lưu kết quả vào: {save_path}")
    
    plt.show()

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️ Sử dụng device: {device}")
    print(f"\n📂 Đọc dữ liệu từ: {args.df_path}")
    df = pd.read_csv(args.df_path)
    print(f"📊 Số lượng records: {len(df)}")
    print(f"📋 Các cột: {df.columns.tolist()}")

    # Xác định base path (thư mục gốc chứa ảnh)
    base_path = args.base_path if hasattr(args, 'base_path') and args.base_path else None
    
    # Lọc ảnh tồn tại trước
    df = filter_existing_images(df, base_path)
    
    if len(df) == 0:
        print("❌ Không có ảnh nào tồn tại! Vui lòng kiểm tra đường dẫn.")
        return

    features_file = "train_with_features.pkl"
    if os.path.exists(features_file):
        print("\n📥 Đang tải DataFrame đã có features từ file...")
        df = pd.read_pickle(features_file)
        print(f"✓ Đã load {len(df)} records với features")
    else:
        print("\n⚙️ File features không tồn tại, sẽ trích xuất mới...")
        df = extract_image_features(df, device=device)
        df.to_pickle(features_file)
        print(f"\n💾 Đã lưu DataFrame với features vào: {features_file}")

    similar_df = find_similar_images(
        query_image_path=args.query_image,
        df=df,
        k=args.k,
        device=device
    )
    
    # Hiển thị kết quả
    print("\n" + "="*60)
    print("🏆 KẾT QUẢ TOP SIMILAR IMAGES")
    print("="*60)
    print(similar_df[['asin', 'title', 'overall', 'similarity_score']].to_string())
    
    # Plot kết quả
    plot_similar_images(
        query_image_path=args.query_image,
        similar_df=similar_df,
        save_path="similar_images_result.png"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find similar images")
    parser.add_argument("--df_path", type=str, required=True, help="Path to the DataFrame CSV file")
    parser.add_argument("--query_image", type=str, required=True, help="Path to the query image")
    parser.add_argument("--k", type=int, default=6, help="Number of similar images to find")
    parser.add_argument("--base_path", type=str, default=None, help="Base path for image files (e.g., RCS_FUSIONDATA/)")
    args = parser.parse_args()
    main(args)