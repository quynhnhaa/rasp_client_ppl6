"""
Multi-Camera Server (ImageZMQ) - Multiprocessing Inference
+ MQTT Publisher for product data
"""

import os
import cv2
import time
import queue
import random
import threading
import numpy as np
import imagezmq
from datetime import datetime
from collections import defaultdict
from multiprocessing import Process, Queue, Manager
import json
from ultralytics import YOLO
import torch
try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False

# ========== Cấu hình ==========
PORT = int(os.getenv("PORT", 5555))
SAVE_DIR = os.getenv("SAVE_DIR", "detections")
INFER_INTERVAL = float(os.getenv("INFER_INTERVAL", 0.15))  # giãn cách giữa các lần inference
NUM_WORKERS = int(os.getenv("NUM_WORKERS", 2))
os.makedirs(SAVE_DIR, exist_ok=True)

# ========== Cấu hình MQTT ==========
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "pbl6/products"

# ========== Thread gửi thông tin sản phẩm qua MQTT ==========
PRODUCTS = {
    "Sting": 10000,
    "Coca": 8000,
    "Pepsi": 8000,
    "Sua": 7000,
    "Banh": 5000
}

def product_sender_mqtt():
    if not MQTT_AVAILABLE:
        print("[ERROR] paho-mqtt library not found. MQTT sender will not start.")
        return

    client = mqtt.Client()
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        print(f"[MQTT Sender] Connected to broker at {MQTT_BROKER}")
    except Exception as e:
        print(f"[ERROR] Could not connect to MQTT broker: {e}")
        return

    client.loop_start()
    print("[MQTT Sender] Started")
    while True:
        try:
            # Sleep for a random interval
            sleep_time = random.randint(5, 15)
            time.sleep(sleep_time)

            # Prepare data
            label = random.choice(list(PRODUCTS.keys()))
            price = PRODUCTS[label]
            quantity = random.randint(1, 5)
            
            message = {
                "label": label,
                "price": price,
                "quantity": quantity
            }
            payload = json.dumps(message)

            # Send data
            result = client.publish(MQTT_TOPIC, payload)
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                print(f"[MQTT Sender] Published: {payload}")
            else:
                print(f"[MQTT Sender] Failed to publish message, return code: {result.rc}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[MQTT Sender] Error: {e}")
            time.sleep(5) # Wait before retrying
    
    client.loop_stop()
    client.disconnect()


# ========== Hàm nhận diện (đã được tích hợp vào worker) ==========


def save_frame(cam_id, frame, boxes):
    if not boxes:
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(SAVE_DIR, f"{cam_id}_{ts}.jpg")
    cv2.imwrite(path, frame)


# ========== Thread A: Nhận ảnh ==========
def receiver(latest_frames, latest_lock, last_infer_ts, in_queue):
    hub = imagezmq.ImageHub(open_port=f"tcp://*:{PORT}")
    print(f"[Receiver] Listening on tcp://*:{PORT}")

    while True:
        try:
            cam_name, jpg_buffer = hub.recv_jpg()
            hub.send_reply(b'OK')
        except Exception as e:
            print(f"[Receiver] Error: {e}")
            time.sleep(0.2)
            continue

        cam_id = cam_name
        np_arr = np.frombuffer(jpg_buffer, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            continue

        with latest_lock:
            latest_frames[cam_id] = frame

        # Đưa frame vào hàng đợi nếu đủ thời gian
        now = time.time()
        if now - last_infer_ts[cam_id] >= INFER_INTERVAL:
            last_infer_ts[cam_id] = now
            if in_queue.full():
                try:
                    in_queue.get_nowait()
                except queue.Empty:
                    pass
            in_queue.put((cam_id, frame.copy()))


# ========== Worker Process: Nhận diện ==========
def worker_process(worker_id, in_q, out_q):
    print(f"[Worker-{worker_id}] Started")
    # device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    model = YOLO("best_conf03_iou045.pt")

    while True:
        cam_id, frame = in_q.get()
        try:
            results = model.predict(
                source=frame,
                conf=0.3,
                iou=0.45,
                device=device,
                verbose=False
            )
            annotated_frame = results[0].plot()
            out_q.put((cam_id, annotated_frame))
        except Exception as e:
            print(f"[Worker-{worker_id}] Error: {e}")


# ========== Thread B: Nhận kết quả từ worker ==========
def result_collector(latest_annotated_frames, latest_lock, out_queue):
    while True:
        try:
            cam_id, annotated_frame = out_queue.get()
            with latest_lock:
                latest_annotated_frames[cam_id] = annotated_frame
        except Exception as e:
            print(f"[Collector] Error processing queue item: {e}")


# ========== Main ==========
def main():
    manager = Manager()
    latest_frames = manager.dict()
    latest_annotated_frames = manager.dict()
    latest_lock = threading.Lock()
    last_infer_ts = defaultdict(lambda: 0.0)
    prev_time = defaultdict(float)
    in_queue = Queue(maxsize=32)
    out_queue = Queue(maxsize=32)

    receiver_args = (latest_frames, latest_lock, last_infer_ts, in_queue)
    threading.Thread(target=receiver, args=receiver_args, daemon=True).start()

    collector_args = (latest_annotated_frames, latest_lock, out_queue)
    threading.Thread(target=result_collector, args=collector_args, daemon=True).start()

    workers = []
    for i in range(NUM_WORKERS):
        p = Process(target=worker_process, args=(i, in_queue, out_queue), daemon=True)
        p.start()
        workers.append(p)
        
    threading.Thread(target=product_sender_mqtt, daemon=True).start()

    print("[Server] Running. Press 'q' to quit.")
    while True:
        items = []
        with latest_lock:
            items = list(latest_frames.items())

        for cam_id, frame in items:
            display_frame = frame.copy()
            
            with latest_lock:
                annotated_frame = latest_annotated_frames.get(cam_id)

            if annotated_frame is not None:
                display_frame = annotated_frame
            
            win = f"Live - {cam_id}"
            cv2.imshow(win, display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    for p in workers:
        p.terminate()


if __name__ == "__main__":
    main()
