# from picamera2 import Picamera2
# import time
# import os

# # 📂 Thư mục lưu ảnh
# SAVE_DIR = "captured_images"
# os.makedirs(SAVE_DIR, exist_ok=True)

# # 📷 Khởi tạo camera
# picam2 = Picamera2()
# config = picam2.create_preview_configuration()
# picam2.configure(config)
# picam2.start()

# print("📸 Camera started. Capturing every 5 frames...")

# frame_count = 0
# image_count = 0

# try:
#     while True:
#         # Chụp frame hiện tại (không lưu)
#         frame = picam2.capture_array()
#         frame_count += 1

#         # Cứ mỗi 5 frame → lưu 1 ảnh
#         # if frame_count % 5 == 0:
#         filename = f"image_{image_count:06d}.jpg"
#         filepath = os.path.join(SAVE_DIR, filename)
#         picam2.capture_file(filepath)
#         print(f"✅ Saved {filename}")
#         image_count += 1

#         # Tốc độ khung hình (điều chỉnh nếu muốn chụp nhanh hơn)
#         time.sleep(0.2)

# except KeyboardInterrupt:
#     print("\n Stopped by user.")
# finally:
#     picam2.stop()
#     print(" Camera stopped.")


from picamera2 import Picamera2
import time, os

SAVE_DIR = "captured_images"
os.makedirs(SAVE_DIR, exist_ok=True)

picam2 = Picamera2()

# 🧠 Cấu hình chụp: main stream là JPEG, raw stream là Bayer (RAW)
config = picam2.create_still_configuration(
    main={"size": (4056, 3040), "format": "BGR888"},  # ảnh màu hiển thị
    raw={"size": (4056, 3040)}  # ảnh RAW cảm biến
)
picam2.configure(config)
picam2.start()

print("📸 Camera started. Capturing RAW + JPEG every 5 frames...")

frame_count = 0
image_count = 0

try:
    while True:
        frame_count += 1
        # nếu Ủn chỉ muốn lưu mỗi 5 frame
        # if frame_count % 5 == 0:
        base = f"image_{image_count:06d}"
        jpeg_path = os.path.join(SAVE_DIR, f"{base}.jpg")
        raw_path = os.path.join(SAVE_DIR, f"{base}.dng")

        # ✨ Lưu ảnh JPEG + RAW
        metadata = picam2.capture_file(jpeg_path)
        picam2.capture_metadata_(raw_path)

        print(f"✅ Saved {base}.jpg & {base}.dng")
        image_count += 1

        time.sleep(0.2)

except KeyboardInterrupt:
    print("\n🛑 Stopped by user.")
finally:
    picam2.stop()
    print("📷 Camera stopped.")
