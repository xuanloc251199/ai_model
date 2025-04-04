import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from functools import partial

# === Cấu hình ===
IMG_SIZE = 128
AGE_CLASSES = [3, 10, 22]  # chỉ còn 3 lớp tuổi
MODEL_PATH = "model_autoencoder_final.h5"

# === Load model ===
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ Chưa tìm thấy model_autoencoder_final.h5")

model = load_model(MODEL_PATH, compile=False)
encoder = model.get_layer("encoder")
decoder = model.get_layer("decoder")

# === Hàm tiện ích ===
def preprocess_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img) / 255.0
    return img_array

def decode_face(latent_vector, age_class_idx):
    age_vector = tf.keras.utils.to_categorical([age_class_idx], num_classes=3)  # đúng số class
    recon = decoder.predict([latent_vector, age_vector])
    recon_img = (recon[0] * 255).astype("uint8")
    return Image.fromarray(recon_img)

def predict_and_render_all(father_path, mother_path, output_label): 
    if not father_path or not mother_path:
        messagebox.showwarning("Thiếu ảnh", "Vui lòng chọn cả ảnh bố và mẹ.")
        return

    try:
        img_f = preprocess_image(father_path)
        img_m = preprocess_image(mother_path)
        z_f = encoder.predict(np.expand_dims(img_f, axis=0))
        z_m = encoder.predict(np.expand_dims(img_m, axis=0))
        z_child = (z_f + z_m) / 2.0

        fig, axes = plt.subplots(1, len(AGE_CLASSES), figsize=(4 * len(AGE_CLASSES), 4))
        for idx, age in enumerate(AGE_CLASSES):
            img_result = decode_face(z_child, idx)
            axes[idx].imshow(img_result)
            axes[idx].axis("off")
            axes[idx].set_title(f"{age} tuổi")

        plt.suptitle("Dự đoán gương mặt bé theo độ tuổi", fontsize=14)
        plt.tight_layout()
        plt.show()
        output_label.config(text="✅ Hiển thị 3 ảnh con thành công!")
    except Exception as e:
        messagebox.showerror("Lỗi", str(e))

# === Giao diện tkinter ===
def create_gui():
    root = tk.Tk()
    root.title("Baby Face Predictor - All Ages")
    root.geometry("500x250")

    father_path = tk.StringVar()
    mother_path = tk.StringVar()

    def browse_file(var):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png")])
        if path:
            var.set(path)

    ttk.Label(root, text="Ảnh bố:").pack(pady=5)
    frame_f = ttk.Frame(root)
    frame_f.pack()
    ttk.Entry(frame_f, textvariable=father_path, width=40).pack(side=tk.LEFT)
    ttk.Button(frame_f, text="Chọn ảnh", command=partial(browse_file, father_path)).pack(side=tk.LEFT)

    ttk.Label(root, text="Ảnh mẹ:").pack(pady=5)
    frame_m = ttk.Frame(root)
    frame_m.pack()
    ttk.Entry(frame_m, textvariable=mother_path, width=40).pack(side=tk.LEFT)
    ttk.Button(frame_m, text="Chọn ảnh", command=partial(browse_file, mother_path)).pack(side=tk.LEFT)

    output_label = ttk.Label(root, text="")
    output_label.pack(pady=10)

    ttk.Button(root, text="🔮 Dự đoán 3 ảnh con", command=lambda: predict_and_render_all(
        father_path.get(), mother_path.get(), output_label
    )).pack(pady=10)

    root.mainloop()

# === Khởi chạy ===
create_gui()
