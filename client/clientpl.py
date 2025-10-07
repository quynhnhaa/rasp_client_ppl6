"""
- Gửi ảnh JPEG chất lượng cao khi có chuyển động
- Gửi heartbeat định kỳ khi không có chuyển động
- Giới hạn FPS để tiết kiệm tài nguyên
- Tự động reconnect nếu mất kết nối
"""

import os
import time
import socket
import cv2
import numpy as np
import imagezmq
from picamera2 import Picamera2
# ========== Cấu hình hệ thống ==========
CONFIG = {
    "server_ip": os.getenv("server_ip", "172.20.10.12"),
    "port": int(os.getenv("port", "5555")),
    "jpeg_quality": int(os.getenv("jpeg_quality", "90")),
    "target_fps": int(os.getenv("target_fps", "10")),
    "frame_size":  list(map(int, os.getenv("frame_size", "640x640").split("x")))
}

CAMERA_NAME = socket.gethostname()
CONNECT_TO = f"tcp://{CONFIG['server_ip']}:{CONFIG['port']}"

# ========== Khởi tạo camera ==========
def init_camera():
    cam = Picamera2()
    cam_config = cam.create_preview_configuration(main = {"size": CONFIG["frame_size"], "format": "RGB888"})
    cam.configure(cam_config)
    cam.start()
    return cam


# ========== Gửi ảnh qua ImageZMQ ==========
def send_frame(sender, frame, quality):
    cam_id = f"{CAMERA_NAME}"
    ok, jpg_buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    jpg_buffer_array = np.array(jpg_buffer).tobytes()
    if ok:
        sender.send_jpg(cam_id, jpg_buffer_array)
# ========== Vòng lặp chính ==========
def main():
    print(f"[INFO] Connecting to server at {CONNECT_TO}")
    sender = None
    backoff = 1.0

    cam = init_camera()
    last_frame_time = time.time()
    frame_interval = 1.0 / CONFIG["target_fps"]

    while True:
        try:
            frame = cam.capture_array()
            
            # Giới hạn FPS
            now = time.time()
            elapsed = now - last_frame_time
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
            last_frame_time = time.time()


            quality = CONFIG["jpeg_quality"] 

            if sender is None:
                try:
                    sender = imagezmq.ImageSender(connect_to=CONNECT_TO, REQ_REP=True)
                    print("[INFO] Connected to server.")
                    backoff = 1.0
                except Exception as e:
                    print(f"[ERROR] Cannot connect: {e}. Retrying in {backoff:.1f}s")
                    time.sleep(backoff)
                    backoff = min(10.0, backoff * 1.5)
                    continue

            try:
                send_frame(sender, frame, quality)
            except Exception as e:
                print(f"[ERROR] Failed to send frame: {e}")
                sender = None
                time.sleep(backoff)
                backoff = min(10.0, backoff * 1.5)

        except KeyboardInterrupt:
            print("[INFO] Interrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"[ERROR] Unexpected error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()
