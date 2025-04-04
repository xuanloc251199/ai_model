import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# === CẤU HÌNH ===
IMG_SIZE = 128
DATA_DIR = "UTKFace_Filtered_Ages"  # Thư mục chứa ảnh lọc theo tuổi
TARGET_AGES = [3, 10, 22]

def load_data():
    images = []
    labels = []

    for age in TARGET_AGES:
        folder = os.path.join(DATA_DIR, str(age))
        if not os.path.isdir(folder):
            continue

        for filename in os.listdir(folder):
            if filename.lower().endswith(".jpg"):
                img_path = os.path.join(folder, filename)
                try:
                    img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
                    img_array = img_to_array(img) / 255.0  # Normalize
                    images.append(img_array)
                    labels.append(age)
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
                    continue

    X = np.array(images)
    y = np.array(labels)

    # Mã hóa nhãn tuổi
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)  # Chuyển 1,3,10,22 → 0,1,2,3
    y_onehot = to_categorical(y_encoded, num_classes=3)

    print(f"Tổng ảnh: {X.shape[0]}, Kích thước mỗi ảnh: {X.shape[1:]}")

    return X, y_onehot, label_encoder

# Dùng thử nếu chạy file độc lập
if __name__ == "__main__":
    X, y_onehot, label_encoder = load_data()
