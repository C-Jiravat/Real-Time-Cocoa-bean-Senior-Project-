import cv2
import numpy as np
import os
from glob import glob

# ตั้งค่าโฟลเดอร์ต้นทางและปลายทาง
input_folder = "/media/gpu2080/Data/Cocoa2025/dataset/color_mix_new_old-20250312T202002Z-001/images/train"        # โฟลเดอร์ภาพต้นฉบับ
output_folder = "/media/gpu2080/Data/Cocoa2025/dataset"

# โฟลเดอร์ย่อย (สร้างอัตโนมัติ)
bright_folder = os.path.join(output_folder, "bright")
sharp_folder = os.path.join(output_folder, "sharp")
bright_sharp_folder = os.path.join(output_folder, "bright_sharp")

os.makedirs(bright_folder, exist_ok=True)
os.makedirs(sharp_folder, exist_ok=True)
os.makedirs(bright_sharp_folder, exist_ok=True)

# ค่าพารามิเตอร์
brightness_value = 20  # เพิ่มความสว่าง +40
sharpen_kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])

# อ่านภาพทั้งหมดในโฟลเดอร์
image_paths = glob(os.path.join(input_folder, "*.*"))

# ประมวลผลทีละรูป
for img_path in image_paths:
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"ไม่สามารถอ่านรูปได้: {img_path}")
        continue

    filename = os.path.basename(img_path)

    # (1) เพิ่มความสว่าง
    bright_img = cv2.convertScaleAbs(img, alpha=1.0, beta=brightness_value)
    cv2.imwrite(os.path.join(bright_folder, filename), bright_img)

    # (2) เพิ่มความคมชัด
    sharp_img = cv2.filter2D(img, -1, sharpen_kernel)
    cv2.imwrite(os.path.join(sharp_folder, filename), sharp_img)

    # (3) เพิ่มทั้งความสว่าง + ความคมชัด
    sharp_then_bright = cv2.convertScaleAbs(sharp_img, alpha=1.0, beta=brightness_value)
    cv2.imwrite(os.path.join(bright_sharp_folder, filename), sharp_then_bright)

    print(f"✔ processed: {filename}")

print("🎉 เสร็จสิ้น! รูปทั้งหมดถูกบันทึกแล้ว")
