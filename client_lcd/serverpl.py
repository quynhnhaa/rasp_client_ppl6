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
PRODUCTS = ["Sting", "Coca", "Pepsi", "Sua", "Banh"]

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
            product_name = random.choice(PRODUCTS)
            total_price = round(random.uniform(10000, 50000))
            message = {"product": product_name, "price": total_price}
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


# ========== Hàm nhận diện (thay bằng model thật) ==========
def detect_products(frame_rgb):
    # Giả lập model: 50% có kết quả, 50% không có
    if random.random() > 0.5:
        h, w = frame_rgb.shape[:2]
        return [(w//4, h//4, w//2, h//2, "product", 0.9)]
    else:
        return []  # Trả về danh sách rỗng khi không phát hiện gì


def draw_boxes(frame, boxes):
    for x1, y1, x2, y2, label, score in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label}:{score:.2f}"
        cv2.putText(frame, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return frame


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
    while True:
        cam_id, frame = in_q.get()
        try:
            # Worker thực hiện nhận diện
            boxes = detect_products(frame)

            # Worker cũng thực hiện vẽ và lưu ảnh để giảm tải cho luồng chính
            if boxes:
                annotated = draw_boxes(frame.copy(), boxes)
                save_frame(cam_id, annotated, boxes)

            # Chỉ gửi lại kết quả (nhẹ) cho luồng chính
            out_q.put((cam_id, boxes))
        except Exception as e:
            print(f"[Worker-{worker_id}] Error: {e}")


# ========== Thread B: Nhận kết quả từ worker ==========
def result_collector(latest_boxes, latest_lock, out_queue):
    while True:
        try:
            cam_id, boxes = out_queue.get()
            # Cập nhật boxes mới nhất cho vòng lặp hiển thị
            with latest_lock:
                latest_boxes[cam_id] = boxes
        except Exception as e:
            print(f"[Collector] Error processing queue item: {e}")


# ========== Main ==========
def main():
    # ========== Bộ nhớ chia sẻ và Queues ==========
    manager = Manager()
    latest_frames = manager.dict()
    latest_boxes = manager.dict()  # Để lưu trữ các bounding box mới nhất
    latest_lock = threading.Lock()
    last_infer_ts = defaultdict(lambda: 0.0)
    in_queue = Queue(maxsize=32)
    out_queue = Queue(maxsize=32)

    # A: receiver thread
    receiver_args = (latest_frames, latest_lock, last_infer_ts, in_queue)
    threading.Thread(target=receiver, args=receiver_args, daemon=True).start()

    # B: collector thread
    collector_args = (latest_boxes, latest_lock, out_queue)
    threading.Thread(target=result_collector, args=collector_args, daemon=True).start()

    # C: inference workers (processes)
    workers = []
    for i in range(NUM_WORKERS):
        p = Process(target=worker_process, args=(i, in_queue, out_queue), daemon=True)
        p.start()
        workers.append(p)
        
    # Start MQTT product sender thread
    threading.Thread(target=product_sender_mqtt, daemon=True).start()

    # D: hiển thị từ main process
    print("[Server] Running. Press 'q' to quit.")
    while True:
        items = []
        with latest_lock:
            # Lấy bản sao của các frame mới nhất
            items = list(latest_frames.items())

        for cam_id, frame in items:
            display_frame = frame.copy()
            
            # Lấy các box mới nhất cho camera này
            with latest_lock:
                boxes = latest_boxes.get(cam_id)

            # Vẽ các box nếu có
            if boxes:
                display_frame = draw_boxes(display_frame, boxes)
            win = f"Live - {cam_id}"
            cv2.imshow(win, display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    for p in workers:
        p.terminate()


if __name__ == "__main__":
    main()