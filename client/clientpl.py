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
from inference import get_model
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
    ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if ok:
        sender.send_jpg(cam_id, jpg)
# ========== Gửi ảnh qua ImageZMQ ==========
def detect_products(frame, model):
    results = model.infer(frame, confidence=0.4, overlap=0.5)[0]
    for pred in results.predictions:
        x, y = int(pred.x), int(pred.y)
        w, h = int(pred.width), int(pred.height)
        conf = pred.confidence
        class_name = pred.class_name
        
        # Tính toạ độ bounding box
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)

        # Vẽ khung chữ nhật
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Vẽ nhãn + confidence
        label = f"{class_name}: {conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame
# ========== Vòng lặp chính ==========
def main():
    print(f"[INFO] Connecting to server at {CONNECT_TO}")
    sender = None
    backoff = 1.0

    cam = init_camera()
    last_frame_time = time.time()
    frame_interval = 1.0 / CONFIG["target_fps"]

    model = model = get_model(
        model_id="vietnamese-productions-classification/19",
        api_key="gMIGrMPVZUiwmUML8smO"
    )

    while True:
        try:
            frame = cam.capture_array()
            frame = detect_products(frame, model)
            
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
