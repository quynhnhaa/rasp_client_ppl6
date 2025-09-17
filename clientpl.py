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
    "server_ip": os.getenv("SERVER_IP", "172.20.10.12"),
    "port": int(os.getenv("PORT", "5555")),
    "jpeg_quality_motion": 85,
    "jpeg_quality_idle": 70,
    "target_fps": 10,
    "heartbeat_interval": 1.0,
    "motion_pixel_threshold": 5000,
    "diff_threshold": 25,
    "blur_kernel": 19, # int(frame_size[0] * 0.03)
    "frame_size": (640, 480)
}

CAMERA_NAME = socket.gethostname()
CONNECT_TO = f"tcp://{CONFIG['server_ip']}:{CONFIG['port']}"

# ========== Khởi tạo camera ==========
def init_camera():
    cam = Picamera2()
    cam_config = cam.create_preview_configuration({"size": CONFIG["frame_size"]})
    cam.configure(cam_config)
    cam.start()
    return cam

# ========== Phát hiện chuyển động ==========
def detect_motion(prev_gray, curr_gray, config):
    if prev_gray is None:
        return False, curr_gray, 0

    diff = cv2.absdiff(prev_gray, curr_gray)
    _, thresh = cv2.threshold(diff, config["diff_threshold"], 255, cv2.THRESH_BINARY)
    motion_pixels = int(np.sum(thresh) / 255)
    motion_detected = motion_pixels > config["motion_pixel_threshold"]
    return motion_detected, curr_gray, motion_pixels

# ========== Gửi ảnh qua ImageZMQ ==========
def send_frame(sender, frame, tag, quality):
    cam_id = f"{CAMERA_NAME}"
    ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if ok:
        sender.send_jpg(cam_id, jpg)

# ========== Vòng lặp chính ==========
def main():
    print(f"[INFO] Connecting to server at {CONNECT_TO}")
    sender = None
    backoff = 1.0

    cam = init_camera()
    prev_gray = None
    last_heartbeat = time.time()
    last_frame_time = time.time()
    frame_interval = 1.0 / CONFIG["target_fps"]

    while True:
        try:
            frame = cam.capture_array()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if CONFIG["blur_kernel"] > 1:
                gray = cv2.GaussianBlur(gray, (CONFIG["blur_kernel"], CONFIG["blur_kernel"]), 0)

            # Giới hạn FPS
            now = time.time()
            elapsed = now - last_frame_time
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
            last_frame_time = time.time()

            # Kiểm tra chuyển động
            motion, prev_gray, score = detect_motion(prev_gray, gray, CONFIG)
            tag = "motion" if motion else "heartbeat"
            quality = CONFIG["jpeg_quality_motion"] if motion else CONFIG["jpeg_quality_idle"]

            # Gửi ảnh nếu có chuyển động hoặc đến thời điểm heartbeat
            if motion or (now - last_heartbeat) >= CONFIG["heartbeat_interval"]:
                if sender is None:
                    try:
                        sender = imagezmq.ImageSender(connect_to=CONNECT_TO, REQ_REP=True)
                        print("[INFO] Connected to server.")
                        # backoff = 1.0
                    except Exception as e:
                        print(f"[ERROR] Cannot connect: {e}. Retrying in {backoff:.1f}s")
                        time.sleep(backoff)
                        backoff = min(10.0, backoff * 1.5)
                        continue

                try:
                    send_frame(sender, frame_rgb, tag, quality)
                    last_heartbeat = now
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
