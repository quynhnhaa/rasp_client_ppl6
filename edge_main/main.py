import os
import csv
import gc
import time
import json
import queue
from threading import Thread
from queue import Queue
import yaml
from collections import Counter
import numpy as np
import paho.mqtt.client as mqtt
try:
    from RPLCD.i2c import CharLCD
    RPLCD_AVAILABLE = True
except ImportError:
    RPLCD_AVAILABLE = False
    print("[WARNING] RPLCD library not found.")

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
        return metadata.get('imgsz', (320, 320))
    except Exception as e:
        print("[ERROR] Metadata:", e)
        return (320, 320)

# ---------------------
# CONFIG
# ---------------------
CONFIG = {
    "model_name": os.getenv("MODEL_NAME", "no_mosaic_sgd_ms_07_ncnn_model"),
    "server_ip": os.getenv("server_ip", "127.0.0.1"),
    "server_port": 5555,
    "camera_name": "raspi_cam",
    "queue_size": 1, 
    "conf_threshold": 0.45,
    "nms_threshold": 0.45,
}

metadata_path = os.path.join(project_dir, CONFIG["model_name"], "metadata.yaml")
CONFIG["camera_resolution"] = load_class_names_from_yaml(metadata_path)
CONFIG["server_address"] = f"tcp://{CONFIG['server_ip']}:{CONFIG['server_port']}"
del metadata_path

# ---------------------
# MQTT & LCD CONFIG
# ---------------------
MQTT_BROKER= "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "pbl6/products"
MQTT_TOPIC_CMD = f"cmd/{CONFIG['camera_name']}"
IS_SCANNING = False
LCD_DISPLAY_DURATION = 3

# ========== Hiển thị trên LCD ==========
def long_string(display, text='', num_line=1, num_cols=16):
    """ 
    Hiển thị chuỗi dài theo kiểu:
    - Nếu chuỗi ngắn hơn num_cols → in thẳng
    - Nếu chuỗi dài hơn num_cols → in 16 ký tự đầu, dừng 1s, rồi cuộn từ PHẢI sang TRÁI
    """
    row = num_line - 1  # RPLCD dùng index bắt đầu từ 0

    if len(text) > num_cols:
        # In 16 ký tự đầu tiên trước
        display.cursor_pos = (row, 0)
        display.write_string(text[:num_cols].ljust(num_cols))
        time.sleep(0.6)

        # Thêm khoảng trắng để cuộn mượt
        scroll_text = text + ' ' * num_cols

        # Cuộn từ phải sang trái
        for i in range(len(scroll_text) - num_cols + 1):
            display.cursor_pos = (row, 0)
            display.write_string(scroll_text[i:i + num_cols])
            time.sleep(0.2)

        time.sleep(1)
    else:
        # Chuỗi ngắn, in thẳng
        display.cursor_pos = (row, 0)
        display.write_string(text.ljust(num_cols))

def display_on_lcd(lcd, label, price, quantity):
    if lcd is None:
        return 0

    time_spent_sleeping = 0
    try:
        # ========== Line 2: Total: {giá tiền} ==========
        lcd.cursor_pos = (1, 0)
        total_price = price * quantity if quantity > 1 else price
        price_str = f"{total_price:,.0f}VND"
        lcd.write_string(price_str.rjust(16)[:16])
        
        # ========== Line 1: <label> x<quantity> ==========
        quantity_str = f" x{quantity}" if quantity > 1 else ""
        quantity_len = len(quantity_str)
        label_cols = 16 - quantity_len

        lcd.cursor_pos = (0, label_cols)
        lcd.write_string(quantity_str)

        text = str(label)
        lcd.cursor_pos = (0, 0)

        if len(text) > label_cols:
            lcd.write_string(text[:label_cols])
            sleep_duration = 0.6
            time.sleep(sleep_duration)
            time_spent_sleeping += sleep_duration

            scroll_text = text + ' ' * label_cols
            for i in range(len(scroll_text) - label_cols + 1):
                lcd.cursor_pos = (0, 0)
                lcd.write_string(scroll_text[i:i + label_cols])
                sleep_duration = 0.1
                time.sleep(sleep_duration)
                time_spent_sleeping += sleep_duration
            
            sleep_duration = 1
            time.sleep(sleep_duration)
            time_spent_sleeping += sleep_duration
        else:
            lcd.write_string(text.ljust(label_cols))

    except Exception as e:
        print(f"[ERROR] Could not write to LCD: {e}")
    
    return time_spent_sleeping

def lcd_worker(q, lcd_obj):
    # Clear screen once at the start
    if lcd_obj:
        lcd_obj.clear()
    # Loop to process incoming messages
    while True:
        item = q.get()
        if item == (None, None, None):  # Sentinel for shutdown
            break
        label, price, quantity = item
        
        # Display item
        time_spent_scrolling = display_on_lcd(lcd_obj, label, price, quantity)
        
        # Wait for the rest of the duration
        remaining_sleep = LCD_DISPLAY_DURATION - time_spent_scrolling
        if remaining_sleep > 0:
            time.sleep(remaining_sleep)

        # Clear screen after displaying
        if lcd_obj:
            lcd_obj.clear()
            
    print("[INFO] LCD worker stopped.")

def init_lcd():
    if not RPLCD_AVAILABLE: return None
    try:
        lcd = CharLCD('PCF8574', 0x27)
        lcd.clear()
        lcd.write_string("Waiting for...")
        print(f"[INFO] LCD initialized at address {hex(0x27)}.")
        return lcd
    except Exception:
        print("[ERROR] Could not initialize LCD.")
        return None

def on_mqtt_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"[MQTT] Connected. Subscribing to {MQTT_TOPIC} and {MQTT_TOPIC_CMD}")
        client.subscribe([(MQTT_TOPIC, 0), (MQTT_TOPIC_CMD, 0)])
    else:
        print(f"[MQTT] Failed to connect, return code {rc}")

def on_mqtt_message(client, userdata, msg):
    global IS_SCANNING
    if msg.topic == MQTT_TOPIC_CMD:
        payload = msg.payload.decode().strip().upper()
        if payload == "SCAN":
            IS_SCANNING = True
            print("[CMD] SCAN STARTED")
        elif payload == "STOP":
            IS_SCANNING = False
            print("[CMD] SCAN STOPPED")
        return

    q = userdata.get('queue')
    try:
        data = json.loads(msg.payload.decode())
        label = data.get("label", "N/A")
        price = data.get("price", 0)
        quantity = data.get("quantity", 1)
        if q:
            q.put((label, price, quantity))
    except Exception as e:
        print(f"[ERROR] MQTT Message: {e}")

# ---------------------
# THREAD 1: CAMERA
# ---------------------
def camera_worker(picam2, frame_queue: Queue):
    while True:
        frame = picam2.capture_array()
        try:
            frame_queue.put(frame, timeout=0.1)
        except queue.Full:
            pass
        del frame

# ---------------------
# THREAD 2: INFERENCE
# ---------------------
def inference_worker(model: YOLO, frame_queue: Queue, sender_frame_queue: Queue):
    global IS_SCANNING
    prev_time = time.time()
    count_gc = 0
    dummy = np.zeros((CONFIG["camera_resolution"][1],CONFIG["camera_resolution"][0],3),dtype=np.uint8)
    model.predict(source=dummy,conf=CONFIG["conf_threshold"],iou=CONFIG["nms_threshold"],verbose=False)
    while True:
        # frame = frame_queue.get()
        try:
            frame = frame_queue.get(timeout=1.0)
        except queue.Empty:
            continue


        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        frame_counter = Counter()
        annotated = None

        if IS_SCANNING:
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
        else:
            annotated = frame
            cv2.putText(annotated, "STOPPED", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Draw FPS
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # SỬA: Dùng put_nowait + try-except để không block
        try:
            sender_frame_queue.put_nowait((annotated, frame_counter, curr_time))
        except queue.Full:
            # Drop frame cũ, put frame mới
            try:
                sender_frame_queue.get_nowait()
            except queue.Empty:
                pass
            sender_frame_queue.put_nowait((annotated, frame_counter, curr_time))

        # --- GIẢI PHÓNG BỘ NHỚ THỦ CÔNG ---
        # Xóa các biến nặng ngay lập tức để tránh OOM trên Pi Zero 2W
        del frame
        if IS_SCANNING:
            del results
            del result
        # annotated đã được put vào queue, sender sẽ lo, ở đây ta xóa tham chiếu cục bộ
        del annotated
        del frame_counter
        
        # Ép chạy Garbage Collection mỗi 30 frame để dọn sạch RAM
        count_gc += 1
        if count_gc > 30:
            gc.collect()
            count_gc = 0

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

    # LCD & MQTT Setup
    lcd_queue = Queue(maxsize=3)
    lcd = init_lcd()
    Thread(target=lcd_worker, args=(lcd_queue, lcd), daemon=True).start()

    mqtt_client = mqtt.Client()
    mqtt_client.user_data_set({'queue': lcd_queue})
    mqtt_client.on_connect = on_mqtt_connect
    mqtt_client.on_message = on_mqtt_message
    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start()
    except Exception as e:
        print(f"[ERROR] MQTT Connection: {e}")

    # Queues
    frame_queue = Queue(maxsize=CONFIG["queue_size"])
    sender_frame_queue = Queue(maxsize=CONFIG["queue_size"])

    # Threads
    Thread(target=camera_worker, args=(picam2, frame_queue), daemon=True).start()
    Thread(target=inference_worker, args=(model, frame_queue, sender_frame_queue), daemon=True).start()

    print("[INFO] System running. Ctrl+C to stop.")

    sender = imagezmq.ImageSender(connect_to=CONFIG["server_address"])
    print(f"[INFO] Connected to server {CONFIG['server_address']}")

    try:
        while True:
            frame, counter, timestamp = sender_frame_queue.get()
            msg = {
                "camera_name": CONFIG["camera_name"],
                "counter": dict(counter),
                "time": timestamp
            }
            sender.send_image(json.dumps(msg), frame)
            del frame
            del counter
    except KeyboardInterrupt:
        mqtt_client.loop_stop()
        if lcd: lcd.close(clear=True)
        picam2.stop()
        print("\n[INFO] Stopped.")