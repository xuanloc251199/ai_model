import os
import shutil

input_dir = "UTKFace_Cropped_Kaggle/"
output_dir = "UTKFace_Filtered_Ages/"
os.makedirs(output_dir, exist_ok=True)

valid_ages = [1, 3, 10, 22]

for age in valid_ages:
    os.makedirs(os.path.join(output_dir, str(age)), exist_ok=True)

processed = 0
skipped = 0

for filename in os.listdir(input_dir):
    if filename.endswith(".jpg"):
        try:
            age = int(filename.split("_")[0])
            if age in valid_ages:
                src_path = os.path.join(input_dir, filename)
                dst_path = os.path.join(output_dir, str(age), filename)
                shutil.copy(src_path, dst_path)
                processed += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"⚠️ Error on {filename}: {e}")
            skipped += 1

print(f"Đã copy {processed} ảnh ở độ tuổi 1, 3, 10, 22.")
print(f"Bỏ qua {skipped} ảnh không thuộc độ tuổi yêu cầu.")
