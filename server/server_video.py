import imagezmq
import cv2
import numpy as np
import time
from datetime import datetime

# ========== Cấu hình ==========
FRAME_SIZE = (640, 640)
FPS = 20
DURATION = 25  # giây
SAVE_DIR = "video_test"

# ========== Tạo file video với timestamp ==========
os.makedirs(SAVE_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
SAVE_PATH = f"{SAVE_DIR}/video_{timestamp}.mp4"

# ========== Khởi tạo ==========
receiver = imagezmq.ImageHub(open_port='tcp://*:5555', REQ_REP=True)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(SAVE_PATH, fourcc, FPS, FRAME_SIZE)

print(f"[INFO] Server is waiting for frames... (saving to {SAVE_PATH})")

frame_count = 0
max_frames = DURATION * FPS
start_time = time.time()

try:
    while True:
        cam_id, jpg_buffer = receiver.recv_jpg()
        receiver.send_reply(b"OK")

        # Giải mã JPEG → frame OpenCV
        np_img = np.frombuffer(jpg_buffer, dtype=np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if frame is None:
            continue

        # Resize cho chắc
        frame = cv2.resize(frame, FRAME_SIZE)

        # Ghi vào video
        if frame_count < max_frames:
            writer.write(frame)
            frame_count += 1

        # Hiển thị trực tiếp
        cv2.imshow("Live Stream from Raspberry Pi", frame)

        # Dừng khi bấm 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Stopped by user.")
            break

        # Dừng sau 25s
        if frame_count >= max_frames:
            print("[INFO] Recorded 25s, stopping recording.")
            break

except KeyboardInterrupt:
    print("[INFO] Interrupted by user.")
finally:
    writer.release()
    cv2.destroyAllWindows()
    receiver.close()
    print(f"[INFO] Video saved to {SAVE_PATH}")
