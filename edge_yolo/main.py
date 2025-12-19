import os
import json
import csv
import gc
import time
import queue
from threading import Thread
from queue import Queue
import yaml
from collections import Counter
from enum import Enum

# MQTT Library
try:
    import paho.mqtt.client as mqtt
except ImportError:
    print("[WARNING] paho-mqtt not installed. Run: pip install paho-mqtt")

# ---------------------
# ENV
# ---------------------
project_dir = os.path.dirname(os.path.abspath(__file__))
os.environ['YOLO_CONFIG_DIR'] = os.path.join(project_dir, '.config')

from ultralytics import YOLO
from picamera2 import Picamera2
import imagezmq
import cv2

# ---------------------
# LOAD METADATA
# ---------------------
def load_class_names_from_yaml(metadata_path):
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = yaml.safe_load(f)
        class_names = [name for _, name in sorted(metadata['names'].items())]
        return class_names, metadata.get('imgsz', (320, 320))
    except Exception as e:
        print("[ERROR] Metadata:", e)
        return ["product"], (320, 320)

def load_price_map(file_path):
    price_map = {}
    try:
        with open(file_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = row.get('label', '').strip()
                price_str = row.get('price', '0').strip()
                if label and price_str:
                    try:
                        price_map[label] = int(price_str)
                    except ValueError:
                        pass
    except Exception as e:
        print(f"[ERROR] Loading prices: {e}")
    return price_map

# ---------------------
# CONFIG
# ---------------------
CONFIG = {
    "model_name": os.getenv("MODEL_NAME", "no_mosaic_sgd_ms_07_ncnn_model"),
    "server_ip": os.getenv("server_ip", "127.0.0.1"),
    "server_port": 5555,
    "camera_name": "raspi_cam",
    "queue_size": 2, # Tăng buffer lên 3 để tránh blocking dây chuyền
    "conf_threshold": 0.45,
    "nms_threshold": 0.45,
}

metadata_path = os.path.join(project_dir, CONFIG["model_name"], "metadata.yaml")
CONFIG["class_names"], CONFIG["camera_resolution"] = load_class_names_from_yaml(metadata_path)
CONFIG["server_address"] = f"tcp://{CONFIG['server_ip']}:{CONFIG['server_port']}"

price_map_path = os.path.join(project_dir, "product_price.csv")
PRICE_MAP = load_price_map(price_map_path)

# ---------------------
# MQTT CONFIG & STATE
# ---------------------
MQTT_BROKER = CONFIG["server_ip"]
MQTT_PORT = 1883

# Tạo topic riêng cho từng Pi dựa trên camera_name
MQTT_TOPIC_CMD = f"cmd/{CONFIG['camera_name']}"   # Server gửi lệnh SCAN/STOP vào đây

SYSTEM_ACTIVE = True      # Mặc định là True để kết nối server lúc đầu
SERVER_MSG = "READY"      # Thông điệp hiển thị

def on_mqtt_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"[MQTT] Connected. Subscribing to: {MQTT_TOPIC_CMD}")
        client.subscribe(MQTT_TOPIC_CMD)
    else:
        print(f"[MQTT] Connection failed: {rc}")

def on_mqtt_message(client, userdata, msg):
    global SYSTEM_ACTIVE, SERVER_MSG
    payload = msg.payload.decode('utf-8').strip()
    
    if msg.topic == MQTT_TOPIC_CMD:
        if payload.upper() == "SCAN":
            SYSTEM_ACTIVE = True
            SERVER_MSG = "SCANNING..."
            print(f"[CMD] START SCANNING on {CONFIG['camera_name']}")
        elif payload.upper() == "STOP":
            SYSTEM_ACTIVE = False
            SERVER_MSG = "STOPPED"
            print(f"[CMD] STOPPED on {CONFIG['camera_name']}")
        elif payload.upper().startswith("TONGTIEN:"):
            money_str = payload.split(":", 1)[1].strip()
            SERVER_MSG = f"TOTAL: {money_str} VND"
            print(SERVER_MSG)

# ---------------------
# FSM
# ---------------------
class ScanState(Enum):
    IDLE = 0
    SCANNING = 1

EMPTY_TIMEOUT = 1.2  # giây

# ---------------------
# THREAD 1: CAMERA
# ---------------------
def camera_worker(picam2, q_raw: Queue):
    while True:
        frame = picam2.capture_array()
        try:
            q_raw.put_nowait(frame)
        except queue.Full:
            pass

# ---------------------
# THREAD 2: INFERENCE
# ---------------------
def inference_worker(model: YOLO, q_raw: Queue, q_data: Queue):
    global SYSTEM_ACTIVE, SERVER_MSG
    prev_time = time.time()
    count_gc = 0
    while True:
        try:
            frame = q_raw.get(timeout=0.1)
        except queue.Empty:
            continue

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        frame_counter = Counter()
        annotated = frame

        # Chỉ chạy AI khi SYSTEM_ACTIVE = True
        if SYSTEM_ACTIVE:
            results = model.predict(
                source=frame,
                conf=CONFIG["conf_threshold"],
                iou=CONFIG["nms_threshold"],
                verbose=False
            )
            result = results[0]

            if result.boxes is not None and len(result.boxes) > 0:
                cls_ids = result.boxes.cls.cpu().numpy().astype(int)
                for cid in cls_ids:
                    frame_counter[result.names[cid]] += 1
            
            annotated = result.plot(font_size=0.4, line_width=1)
            del results, result
        else:
            # Nếu dừng, vẽ thông báo chờ
            cv2.putText(annotated, "WAITING FOR CMD...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

        # Draw FPS
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        # Draw Server Message
        cv2.putText(annotated, SERVER_MSG, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Đóng gói dữ liệu để chuyển sang FSM
        packet = {
            "frame": annotated,
            "counter": frame_counter,
            "time": time.time()
        }

        try:
            q_data.put_nowait(packet)
        except queue.Full:
            pass

        # --- GIẢI PHÓNG BỘ NHỚ THỦ CÔNG ---
        # Xóa các biến nặng ngay lập tức để tránh OOM trên Pi Zero 2W
        del frame
        # annotated đã được put vào queue, sender sẽ lo, ở đây ta xóa tham chiếu cục bộ
        del annotated
        
        # Ép chạy Garbage Collection mỗi 30 frame để dọn sạch RAM
        count_gc += 1
        if count_gc > 30:
            gc.collect()
            count_gc = 0

# ---------------------
# THREAD 3: SCAN FSM
# ---------------------
def scan_fsm_worker(q_data: Queue, q_result: Queue):
    state = ScanState.IDLE
    last_seen_time = 0
    batch_frame_counters = []
    session_total = Counter()  # Mục "đã quét" (tổng các đợt)
    current_scanning = {}    # Mục "đang quét" (đợt hiện tại)
    while True:
        try:
            data = q_data.get(timeout=0.1)
        except queue.Empty:
            continue

        frame = data["frame"]
        frame_counter = data["counter"]
        now = data["time"]

        # Nếu hệ thống dừng, reset trạng thái FSM nhưng VẪN GỬI FRAME đi tiếp
        if not SYSTEM_ACTIVE:
            if state != ScanState.IDLE or session_total:
                state = ScanState.IDLE
                last_seen_time = 0
                batch_frame_counters = []
                session_total = Counter() 
                current_scanning = {}
                print("[FSM] Session Reset (STOPPED)")
            
            # Bỏ qua logic FSM, chuyển thẳng frame xuống Sender
            try:
                q_result.put_nowait({
                    "frame": frame,
                    "current": {},
                    "total": {}
                })
            except queue.Full: pass
            continue

        if state == ScanState.IDLE:
            if frame_counter:
                state = ScanState.SCANNING
                batch_frame_counters = [frame_counter]
                last_seen_time = now
                print("\n[SCAN] START")

        elif state == ScanState.SCANNING:
            if frame_counter:
                batch_frame_counters.append(frame_counter)
                last_seen_time = now
            
            # Cập nhật mục "đang quét" từ dữ liệu batch hiện có
            if batch_frame_counters:
                votes = Counter(tuple(sorted(c.items())) for c in batch_frame_counters)
                if votes:
                    best, _ = votes.most_common(1)[0]
                    current_scanning = dict(best)

            if not frame_counter:
                if now - last_seen_time > EMPTY_TIMEOUT:
                    # Điều kiện thoả mãn: Kết thúc đợt -> Chuyển từ "đang quét" sang "đã quét"
                    if current_scanning:
                        if len(batch_frame_counters) < 5:
                            print(f"[SCAN] IGNORED (Too few frames: {len(batch_frame_counters)})")
                        else:
                            print("[SCAN] END →", current_scanning)
                            session_total.update(current_scanning)
                            
                            # Tính tổng tiền dựa trên PRICE_MAP đã load
                            total_money = sum(PRICE_MAP.get(k, 0) * v for k, v in session_total.items())
                            print(f"[SESSION TOTAL] → {dict(session_total)} | Money: {total_money:,} VND")

                        current_scanning = {}

                    state = ScanState.IDLE
                    batch_frame_counters = []
                # else: Điều kiện sai -> Vẫn giữ ở mục "đang quét" (current_scanning)

        # Put kết quả với timeout để tránh treo luồng FSM
        try:
            q_result.put_nowait({
                "frame": frame,
                "current": current_scanning,
                "total": dict(session_total)
            })
        except queue.Full: pass

# ---------------------
# THREAD 4: SENDER
# ---------------------
def sender_worker(q_result: Queue, server_address: str, camera_name: str):
    sender = imagezmq.ImageSender(connect_to=server_address)
    print(f"[INFO] Connected to server {server_address}")

    while True:
        data = q_result.get()
        frame = data["frame"]
        
        msg = {
            "camera_name": camera_name,
            "current": data["current"],
            "total": data["total"]
        }
        sender.send_image(json.dumps(msg), frame)

def startup_worker():
    global SYSTEM_ACTIVE, SERVER_MSG
    print("[INFO] Startup: Active for 5s to connect server...")
    time.sleep(10)
    if SERVER_MSG != "SCANNING...":
        SYSTEM_ACTIVE = False
        SERVER_MSG = "STOPPED"
        print("[INFO] Startup finished. System inactive.")

# ---------------------
# MAIN
# ---------------------
if __name__ == "__main__":
    print("=" * 50)
    print("YOLO NCNN + FSM SCAN PIPELINE")
    print("=" * 50)

    # Camera
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": CONFIG["camera_resolution"], "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()

    # Model
    model = YOLO(CONFIG["model_name"], task="detect")

    # Queues
    q_raw = Queue(maxsize=CONFIG["queue_size"])    # Camera -> Inference
    q_data = Queue(maxsize=CONFIG["queue_size"])   # Inference -> FSM
    q_result = Queue(maxsize=CONFIG["queue_size"]) # FSM -> Sender

    # MQTT Client Start
    if 'mqtt' in globals():
        mqtt_client = mqtt.Client()
        mqtt_client.on_connect = on_mqtt_connect
        mqtt_client.on_message = on_mqtt_message
        print(f"[INFO] Connecting MQTT to {MQTT_BROKER}...")
        try:
            mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
            mqtt_client.loop_start()
        except Exception as e:
            print(f"[ERROR] MQTT Connection failed: {e}")
            print(f"[HINT] Kiểm tra Mosquitto trên Server đã có 'listener 1883' và 'allow_anonymous true' chưa?")

    # Threads
    Thread(target=camera_worker, args=(picam2, q_raw), daemon=True).start()
    Thread(target=inference_worker, args=(model, q_raw, q_data), daemon=True).start()
    Thread(target=scan_fsm_worker, args=(q_data, q_result), daemon=True).start()
    Thread(target=sender_worker, args=(q_result, CONFIG["server_address"], CONFIG["camera_name"]), daemon=True).start()
    Thread(target=startup_worker, daemon=True).start()

    print("[INFO] System running. Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        if 'mqtt_client' in locals():
            mqtt_client.loop_stop()
        picam2.stop()
        print("\n[INFO] Stopped.")
