import imagezmq
import cv2
import time
import numpy as np

# ========== Cấu hình ==========
SAVE_PATH = "video_test1.mp4"
FRAME_SIZE = (640, 640)
FPS = 20
DURATION = 25  # giây

# ========== Khởi tạo ==========
receiver = imagezmq.ImageHub(open_port='tcp://*:5555', REQ_REP=True)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(SAVE_PATH, fourcc, FPS, FRAME_SIZE)

print("[INFO] Server is waiting for frames...")

frame_count = 0
max_frames = DURATION * FPS
start_time = time.time()

try:
    while True:
        cam_id, jpg_buffer = receiver.recv_jpg()
        receiver.send_reply(b"OK")

        # Giải mã ảnh JPEG thành frame
        np_img = np.frombuffer(jpg_buffer, dtype=np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # Resize nếu không đúng kích thước
        frame = cv2.resize(frame, FRAME_SIZE)

        # Ghi vào file video
        writer.write(frame)
        frame_count += 1

        # Hiển thị cho debug (có thể tắt)
        cv2.imshow("Receiving", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Dừng sau 25s
        if frame_count >= max_frames:
            print("[INFO] Reached 25s video length, stopping recording.")
            break

except KeyboardInterrupt:
    print("[INFO] Interrupted by user.")
finally:
    writer.release()
    cv2.destroyAllWindows()
    receiver.close()
    print(f"[INFO] Video saved to {SAVE_PATH}")
