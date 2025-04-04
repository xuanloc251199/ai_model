import base64

def image_to_base64(image_path: str) -> str:
    """
    Chuyển ảnh thành chuỗi base64 mà không có tiền tố data:image/jpeg;base64.
    """
    with open(image_path, "rb") as image_file:
        # Đọc ảnh và chuyển sang chuỗi base64
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    
    return encoded_string

# Thử chuyển ảnh thành base64
image_path = 'results/generated_baby.jpg'  # Đổi đường dẫn này thành đường dẫn tới ảnh của bạn
encoded_image = image_to_base64(image_path)
print('result:' + encoded_image)  # In ra chuỗi base64 không có tiền tố
