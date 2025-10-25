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



IMG_SIZE = 224

def load_image_tensor(file_path: str) -> torch.Tensor:
    path = Path(file_path)
    if not path.exists():
        print(f"Ảnh không tồn tại: {file_path}")
        return torch.zeros(3, IMG_SIZE, IMG_SIZE)
    
    try:
        with Image.open(path).convert('RGB') as im:
            return img_tf(im)
    except Exception as e:
        print(f"Lỗi khi load ảnh {file_path}: {e}")
        return torch.zeros(3, IMG_SIZE, IMG_SIZE)

def extract_image_features(df: pd.DataFrame, device='cuda') -> pd.DataFrame:
    img_enc = ImageEncoder().to(device)
    img_enc.eval()
    
    features_list = []
    
    print(f"Đang trích xuất features cho {len(df)} ảnh...")
    with torch.no_grad():
        for idx, row in df.iterrows():
            if idx % 100 == 0:
                print(f"Đã xử lý {idx}/{len(df)} ảnh...")
            img_tensor = load_image_tensor(row['file_path'])
            img_batch = img_tensor.unsqueeze(0).to(device)
            feature = img_enc(img_batch)  # (1, 4096)
            feature_np = feature.cpu().numpy().flatten()
            features_list.append(feature_np)
    df['img_feature'] = features_list
    print(f"Hoàn thành trích xuất features!")
    df.to_pickle("train_with_features.pkl")
    return df

def find_similar_images(query_image_path: str, df: pd.DataFrame, k=6, device='cuda'):
    print(f"\nĐang tìm {k} ảnh giống với: {query_image_path}")
    
    # Trích xuất feature của ảnh query
    img_enc = ImageEncoder().to(device)
    img_enc.eval()
    
    with torch.no_grad():
        query_tensor = load_image_tensor(query_image_path)
        query_batch = query_tensor.unsqueeze(0).to(device)
        query_feature = img_enc(query_batch).cpu().numpy().flatten()
    
    # Lấy ASIN của ảnh query (nếu có trong DataFrame)
    query_asin = None
    query_match = df[df['file_path'] == query_image_path]
    if not query_match.empty:
        query_asin = query_match.iloc[0]['asin']
        print(f"ASIN của ảnh query: {query_asin}")
    
    features_matrix = np.vstack(df['img_feature'].values)
    
    print("Đang tìm kiếm bằng KNN...")
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
        print(f"Đã lọc bỏ {len(df.iloc[indices[0]]) - len(similar_df)} ảnh có cùng ASIN: {query_asin}")
    
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
    print(f"Sử dụng device: {device}")
    print(f"\nĐọc dữ liệu từ: {args.df_path}")
    df = pd.read_csv(args.df_path)
    df['file_path'] = df['file_path']    
    print(f"Số lượng records: {len(df)}")
    print(f"Các cột: {df.columns.tolist()}")

    features_file = "train_with_features.pkl"
    if os.path.exists(features_file):
        print("\nĐang tải DataFrame đã có features từ file...")
        df = pd.read_pickle(features_file)
    else:
        print("\nFile features không tồn tại, sẽ trích xuất mới...")
        if 'img_feature' not in df.columns:
            df = extract_image_features(df, device=device)
            output_path = features_file
            df.to_pickle(output_path)
            print(f"\nĐã lưu DataFrame với features vào: {output_path}")
        else:
            print("\nDataFrame đã có cột 'img_feature', bỏ qua bước trích xuất")
    

    similar_df = find_similar_images(
        query_image_path=args.query_image,
        df=df,
        k=args.k,
        device=device
    )
    
    # 4. Hiển thị kết quả
    print("\n=== KẾT QUẢ TOP SIMILAR IMAGES ===")
    print(similar_df[['asin', 'title', 'overall', 'similarity_score']].to_string())
    
    # Plot kết quả
    plot_similar_images(
        query_image_path=args.query_image,
        similar_df=similar_df,
        save_path="similar_images_result.png"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find similar images")
    parser.add_argument("--df_path", type=str, required=True, help="Path to the DataFrame")
    parser.add_argument("--query_image", type=str, required=True, help="Path to the query image")
    parser.add_argument("--k", type=int, default=6, help="Number of similar images to find")
    args = parser.parse_args()
    main(args)