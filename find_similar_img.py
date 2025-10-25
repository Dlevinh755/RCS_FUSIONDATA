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

def extract_image_features(df: pd.DataFrame, device='cuda', cache_file=None) -> pd.DataFrame:
    """
    Trích xuất feature từ tất cả ảnh trong DataFrame
    
    Args:
        df: DataFrame có cột 'file_path'
        device: 'cuda' hoặc 'cpu'
        cache_file: Đường dẫn file để cache kết quả
    
    Returns:
        DataFrame với cột mới 'img_feature'
    """
    print("Đang khởi tạo ImageEncoder...")
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
    
    # Lưu cache nếu được chỉ định
    if cache_file:
        df.to_pickle(cache_file)
        print(f"Đã lưu cache vào: {cache_file}")
    
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
    
    # Chuẩn hóa đường dẫn để so sánh
    query_path_normalized = str(Path(query_image_path).resolve())
    
    # Lấy ASIN của ảnh query (nếu có trong DataFrame)
    query_asin = None
    for idx, row in df.iterrows():
        if str(Path(row['file_path']).resolve()) == query_path_normalized:
            query_asin = row['asin']
            print(f"ASIN của ảnh query: {query_asin}")
            break
    
    # Chuẩn bị dữ liệu cho KNN
    features_matrix = np.vstack(df['img_feature'].values)
    
    print("Đang tìm kiếm bằng KNN...")
    # Tìm nhiều hơn k ảnh để đảm bảo đủ sau khi lọc
    search_k = min(k * 5, len(df))  # Tăng từ k*3 lên k*5 để an toàn hơn
    knn = NearestNeighbors(n_neighbors=search_k, metric='cosine')
    knn.fit(features_matrix)
    distances, indices = knn.kneighbors([query_feature])
    
    # Tạo DataFrame kết quả
    similar_df = df.iloc[indices[0]].copy()
    similar_df['distance'] = distances[0]
    similar_df['similarity_score'] = 1 - distances[0]
    
    # Lọc bỏ ảnh query và các ảnh có cùng ASIN
    original_count = len(similar_df)
    
    # Loại bỏ chính ảnh query
    similar_df = similar_df[
        similar_df['file_path'].apply(lambda x: str(Path(x).resolve())) != query_path_normalized
    ]
    
    # Loại bỏ các ảnh có cùng ASIN (nếu tìm thấy)
    if query_asin:
        similar_df = similar_df[similar_df['asin'] != query_asin]
        filtered_count = original_count - len(similar_df)
        print(f"Đã lọc bỏ {filtered_count} ảnh (bao gồm ảnh query và các ảnh cùng ASIN: {query_asin})")
    
    # Lấy top k kết quả
    similar_df = similar_df.head(k)
    
    if len(similar_df) < k:
        print(f"⚠️ Cảnh báo: Chỉ tìm được {len(similar_df)}/{k} ảnh sau khi lọc")
    
    return similar_df

def plot_similar_images(query_image_path: str, similar_df: pd.DataFrame, save_path=None):
    """
    Hiển thị ảnh query và các ảnh giống nhất
    
    Args:
        query_image_path: Đường dẫn ảnh query
        similar_df: DataFrame chứa các ảnh giống
        save_path: Đường dẫn để lưu hình (optional)
    """
    n_results = len(similar_df)
    # Tính số hàng cần thiết
    n_cols = 3
    n_rows = (n_results + n_cols) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    # Hiển thị ảnh query
    try:
        query_img = Image.open(query_image_path).convert('RGB')
        axes[0].imshow(query_img)
        axes[0].set_title('🔍 Query Image', fontsize=14, fontweight='bold', color='blue')
        axes[0].axis('off')
    except Exception as e:
        print(f"Lỗi khi hiển thị ảnh query: {e}")
        axes[0].text(0.5, 0.5, 'Query Image\nError', ha='center', va='center')
        axes[0].axis('off')
    
    # Hiển thị các ảnh giống nhất
    for idx, (_, row) in enumerate(similar_df.iterrows(), start=1):
        if idx >= len(axes):
            break
            
        try:
            img = Image.open(row['file_path']).convert('RGB')
            axes[idx].imshow(img)
            
            # Tạo title với thông tin chi tiết
            title = f"#{idx} - Similarity: {row['similarity_score']:.3f}\n"
            title += f"⭐ Rating: {row['overall']:.1f}\n"
            title += f"📦 ASIN: {row['asin']}\n"
            # Cắt ngắn title nếu quá dài
            product_title = str(row.get('title', 'N/A'))
            if len(product_title) > 40:
                product_title = product_title[:37] + '...'
            title += f"📝 {product_title}"
            
            axes[idx].set_title(title, fontsize=9)
            axes[idx].axis('off')
        except Exception as e:
            print(f"Lỗi khi hiển thị ảnh {row['file_path']}: {e}")
            axes[idx].text(0.5, 0.5, f'Image #{idx}\nError', ha='center', va='center')
            axes[idx].axis('off')
    
    # Tắt các subplot thừa
    for idx in range(len(similar_df) + 1, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Đã lưu kết quả vào: {save_path}")
    
    plt.show()

def main(args):
    """Hàm chính để chạy demo"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Sử dụng device: {device}")
    
    # Đọc dữ liệu
    print(f"\n📂 Đọc dữ liệu từ: {args.df_path}")
    df = pd.read_csv(args.df_path)
    
    # Xử lý đường dẫn ảnh
    if args.data_path:
        data_path = Path(args.data_path)
        print(f"📁 Data path: {data_path}")
        # Cập nhật file_path thành đường dẫn tuyệt đối
        df['file_path'] = df['file_path'].apply(lambda x: str(data_path / x) if not Path(x).is_absolute() else x)
    
    print(f"📊 Số lượng records: {len(df)}")
    print(f"📋 Các cột: {df.columns.tolist()}")

    # Xử lý features
    features_file = args.cache_file if args.cache_file else "train_with_features.pkl"
    
    if os.path.exists(features_file):
        print(f"\n💾 Đang tải DataFrame đã có features từ: {features_file}")
        df_cached = pd.read_pickle(features_file)
        
        # Kiểm tra xem cache có đủ dữ liệu không
        if len(df_cached) >= len(df):
            df = df_cached
            print("✅ Đã load features từ cache")
        else:
            print("⚠️  Cache không đầy đủ, sẽ trích xuất lại")
            df = extract_image_features(df, device=device, cache_file=features_file)
    else:
        print(f"\n🔄 File features không tồn tại, đang trích xuất mới...")
        df = extract_image_features(df, device=device, cache_file=features_file)
    
    # Kiểm tra query image có tồn tại không
    query_path = Path(args.query_image)
    if not query_path.exists():
        # Thử ghép với data_path nếu là đường dẫn tương đối
        if args.data_path:
            query_path = Path(args.data_path) / args.query_image
            if not query_path.exists():
                raise FileNotFoundError(f"❌ Không tìm thấy ảnh query: {args.query_image}")
        else:
            raise FileNotFoundError(f"❌ Không tìm thấy ảnh query: {args.query_image}")
    
    print(f"\n🔍 Query image: {query_path}")
    
    # Tìm ảnh giống
    similar_df = find_similar_images(
        query_image_path=str(query_path),
        df=df,
        k=args.k,
        device=device
    )
    
    # Hiển thị kết quả
    print("\n" + "="*80)
    print("🎯 KẾT QUẢ TOP SIMILAR IMAGES")
    print("="*80)
    display_cols = ['asin', 'title', 'overall', 'similarity_score']
    # Chỉ hiển thị các cột có trong DataFrame
    display_cols = [col for col in display_cols if col in similar_df.columns]
    print(similar_df[display_cols].to_string(index=False))
    print("="*80)
    
    # Plot kết quả
    output_path = args.output if args.output else "similar_images_result.png"
    plot_similar_images(
        query_image_path=str(query_path),
        similar_df=similar_df,
        save_path=output_path
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="🔍 Tìm ảnh tương tự sử dụng ImageEncoder và KNN")
    parser.add_argument("--df_path", type=str, required=True, 
                        help="Đường dẫn đến file CSV chứa dữ liệu")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Đường dẫn gốc chứa thư mục images/ (nếu file_path trong CSV là relative path)")
    parser.add_argument("--query_image", type=str, required=True, 
                        help="Đường dẫn đến ảnh cần tìm kiếm")
    parser.add_argument("--k", type=int, default=6, 
                        help="Số lượng ảnh tương tự cần tìm (default: 6)")
    parser.add_argument("--cache_file", type=str, default=None,
                        help="Đường dẫn file cache cho features (default: train_with_features.pkl)")
    parser.add_argument("--output", type=str, default=None,
                        help="Đường dẫn file output để lưu kết quả (default: similar_images_result.png)")
    
    args = parser.parse_args()
    main(args)