import os
import csv
import gc
import time
import json
import queue
from threading import Thread
from multiprocessing import Process, Queue as MPQueue, Event, Value
import yaml
from collections import Counter
import numpy as np
import paho.mqtt.client as mqtt
import zmq
import signal
import sys

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

# NOTE: Import YOLO trong process con, không import ở đây
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
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "pbl6/products"
MQTT_TOPIC_CMD = f"cmd/{CONFIG['camera_name']}"
LCD_DISPLAY_DURATION = 1.5

# ---------------------
# LCD FUNCTIONS
# ---------------------
def display_on_lcd(lcd, label, price, quantity):
    if lcd is None:
        return 0
    try:
        lcd.cursor_pos = (1, 0)
        total_price = price * quantity if quantity > 1 else price
        price_str = f"{total_price:,.0f}VND"
        lcd.write_string(price_str.rjust(16)[:16])

        quantity_str = f" x{quantity}" if quantity > 1 else ""
        quantity_len = len(quantity_str)
        label_cols = 16 - quantity_len

        lcd.cursor_pos = (0, label_cols)
        lcd.write_string(quantity_str)

        text = str(label)
        lcd.cursor_pos = (0, 0)
        if len(text) > label_cols:
            lcd.write_string(text[:label_cols])
        else:
            lcd.write_string(text.ljust(label_cols))
    except Exception as e:
        print(f"[ERROR] LCD: {e}")
    return 0

def lcd_worker(q, lcd_obj):
    if lcd_obj:
        lcd_obj.clear()
    while True:
        item = q.get()
        if item == (None, None, None):
            break
        label, price, quantity = item
        display_on_lcd(lcd_obj, label, price, quantity)
        time.sleep(LCD_DISPLAY_DURATION)
        if lcd_obj:
            lcd_obj.clear()
    print("[INFO] LCD worker stopped.")

def init_lcd():
    if not RPLCD_AVAILABLE:
        return None
    try:
        lcd = CharLCD('PCF8574', 0x27)
        lcd.clear()
        lcd.write_string("Waiting for...")
        print(f"[INFO] LCD initialized.")
        return lcd
    except Exception:
        print("[ERROR] Could not initialize LCD.")
        return None

# ---------------------
# MQTT CALLBACKS
# ---------------------
def on_mqtt_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"[MQTT] Connected. Subscribing to {MQTT_TOPIC} and {MQTT_TOPIC_CMD}")
        client.subscribe([(MQTT_TOPIC, 0), (MQTT_TOPIC_CMD, 0)])
    else:
        print(f"[MQTT] Failed to connect, return code {rc}")

def on_mqtt_message(client, userdata, msg):
    # Lấy Event từ userdata
    scanning_event = userdata.get('scanning_event')
    lcd_queue = userdata.get('lcd_queue')

    if msg.topic == MQTT_TOPIC_CMD:
        payload = msg.payload.decode().strip().upper()
        if payload == "SCAN":
            if scanning_event:
                scanning_event.set()
            print("[CMD] SCAN STARTED")
        elif payload == "STOP":
            if scanning_event:
                scanning_event.clear()
            print("[CMD] SCAN STOPPED")
        return

    # Handle product messages for LCD
    try:
        data = json.loads(msg.payload.decode())
        label = data.get("label", "N/A")
        price = data.get("price", 0)
        quantity = data.get("quantity", 1)
        if lcd_queue:
            try:
                lcd_queue.put_nowait((label, price, quantity))
            except queue.Full:
                try:
                    lcd_queue.get_nowait()
                except queue.Empty:
                    pass
                lcd_queue.put_nowait((label, price, quantity))
    except Exception as e:
        print(f"[ERROR] MQTT Message: {e}")

# ---------------------
# Non-blocking Queue với drop policy
# ---------------------
def safe_queue_put(q, item, max_retries=2):
    """Put item vào queue, drop old item nếu full"""
    for _ in range(max_retries):
        try:
            q.put_nowait(item)
            return True
        except:
            # Queue full → drop oldest
            try:
                old = q.get_nowait()
                del old
            except:
                pass
    return False

def safe_queue_get(q, timeout=0.1):
    """Get item từ queue với timeout"""
    try:
        return q.get(timeout=timeout), True
    except queue.Empty:
        return None, False

# ---------------------
# CAMERA WORKER (đã sửa)
# ---------------------
def camera_worker(picam2, frame_queue: MPQueue, stop_event: Event, cam_heartbeat: Value):
    print("[CAMERA] Worker started.")
    frame_count = 0
    
    while not stop_event.is_set():
        try:
            frame = picam2.capture_array()
            cam_heartbeat.value = time.time()  # Cập nhật nhịp tim camera
            
            # Non-blocking put với drop policy
            if not safe_queue_put(frame_queue, frame):
                pass  # Drop frame nếu không put được
            
            del frame
            frame_count += 1
            
            # GC định kỳ
            if frame_count % 50 == 0:
                gc.collect()
                
        except Exception as e:
            print(f"[ERROR] Camera: {e}")
            time.sleep(0.1)
            
    print("[CAMERA] Worker stopped.")

# ---------------------
# INFERENCE PROCESS 
# ---------------------
def inference_process(model_name: str,model_config: dict,frame_queue: MPQueue,
                      result_queue: MPQueue,scanning_event: Event,stop_event: Event):
    from ultralytics import YOLO
    from collections import Counter
    
    print("[INFERENCE] Process started.")
    
    # Load model
    model = YOLO(model_name, task="detect")
    
    # Warmup
    dummy = np.zeros(
        (model_config["resolution"][1], model_config["resolution"][0], 3),
        dtype=np.uint8
    )
    model.predict(source=dummy, verbose=False)
    del dummy
    gc.collect()
    print("[INFERENCE] Warmup complete.")
    
    prev_time = time.time()
    gc_counter = 0
    dropped_frames = 0
    
    while not stop_event.is_set():
        # ===== Get frame (non-blocking) =====
        frame, ok = safe_queue_get(frame_queue, timeout=0.5)
        if not ok:
            continue
        
        # ===== Drain queue - chỉ giữ frame mới nhất =====
        while True:
            try:
                newer_frame = frame_queue.get_nowait()
                del frame  # Xóa frame cũ
                frame = newer_frame
                dropped_frames += 1
            except:
                break
        
        # ===== FPS =====
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        
        # ===== Inference =====
        frame_counter = Counter()
        is_scanning = scanning_event.is_set()
        
        try:
            if is_scanning:
                results = model.predict(
                    source=frame,
                    conf=model_config["conf_threshold"],
                    iou=model_config["nms_threshold"],
                    verbose=False
                )
                result = results[0]
                
                if result.boxes is not None and len(result.boxes) > 0:
                    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
                    for cid in cls_ids:
                        frame_counter[result.names[cid]] += 1
                    del cls_ids
                
                annotated = result.plot(font_size=0.4, line_width=1)
                del result, results
            else:
                annotated = frame.copy()
                # annotated = frame
                cv2.putText(annotated, "STOPPED", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        except Exception as e:
            print(f"[ERROR] Inference: {e}")
            annotated = frame
        
        del frame
        
        # ===== Draw FPS =====
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # ===== Put result (non-blocking) =====
        result_data = (annotated, dict(frame_counter), curr_time)
        safe_queue_put(result_queue, result_data)
        
        # ===== Cleanup =====
        del annotated, frame_counter
        
        gc_counter += 1
        if gc_counter >= 30:
            gc.collect()
            gc_counter = 0
            if dropped_frames > 0:
                # print(f"[INFERENCE] Dropped {dropped_frames} frames")
                dropped_frames = 0
    
    print("[INFERENCE] Process stopped.")

# ---------------------
# MAIN (đã sửa)
# ---------------------
if __name__ == "__main__":
    print("=" * 50)
    print("YOLO NCNN + MULTIPROCESSING (FIXED)")
    print("=" * 50)
    
    # Events
    scanning_event = Event()
    stop_event = Event()

    def signal_handler(sig, frame):
        # Cho phép bấm Ctrl+C lần 2 để force exit nếu bị treo
        if stop_event.is_set():
            print("\n[INFO] Force exit triggered...")
            os._exit(1)
        print("\n[INFO] Ctrl+C detected. Stopping...")
        stop_event.set()
    signal.signal(signal.SIGINT, signal_handler)
    
    # Camera
    picam2 = Picamera2()
    cam_config = picam2.create_preview_configuration(
        main={"size": CONFIG["camera_resolution"], "format": "RGB888"}
    )
    picam2.configure(cam_config)
    picam2.start()
    
    # Queues 
    frame_queue = MPQueue(maxsize=1)
    result_queue = MPQueue(maxsize=1)
    
    # Shared Value để theo dõi trạng thái Camera
    cam_heartbeat = Value('d', time.time())
    
    # LCD Setup
    lcd_queue = queue.Queue(maxsize=3)
    lcd = init_lcd()
    Thread(target=lcd_worker, args=(lcd_queue, lcd), daemon=True).start()

    # MQTT Setup
    mqtt_client = mqtt.Client()
    mqtt_client.user_data_set({
        'scanning_event': scanning_event,
        'lcd_queue': lcd_queue
    })
    mqtt_client.on_connect = on_mqtt_connect
    mqtt_client.on_message = on_mqtt_message
    
    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start()
    except Exception as e:
        print(f"[ERROR] MQTT Connection: {e}")

    # Config
    model_config = {
        "resolution": CONFIG["camera_resolution"],
        "conf_threshold": CONFIG["conf_threshold"],
        "nms_threshold": CONFIG["nms_threshold"],
    }
    
    # Start workers
    camera_thread = Thread(
        target=camera_worker,
        args=(picam2, frame_queue, stop_event, cam_heartbeat),
        daemon=True
    )
    camera_thread.start()
    
    inference_proc = Process(
        target=inference_process,
        args=(
            CONFIG["model_name"],
            model_config,
            frame_queue,
            result_queue,
            scanning_event,
            stop_event
        )
    )
    inference_proc.start()
    
    # ===== ImgageZMQ cho gửi frame =====
    sender = imagezmq.ImageSender(connect_to=CONFIG["server_address"],REQ_REP=True)
    # Set timeout để không bị treo khi server chết (2000ms = 2s)
    sender.zmq_socket.setsockopt(zmq.RCVTIMEO, 2000)
    sender.zmq_socket.setsockopt(zmq.LINGER, 0)
    print(f"[INFO] Connected to server (PUB-SUB mode)")
    
    # Main loop với timeout
    print("[INFO] System running. Ctrl+C to stop.")
    
    last_frame_time = time.time()
    watchdog_timeout = 5.0  # 5 giây không có frame -> cảnh báo
    
    try:
        while not stop_event.is_set():
            # Get result với timeout
            data, ok = safe_queue_get(result_queue, timeout=1.0)
            
            if ok:
                frame, counter, timestamp = data
                last_frame_time = time.time()
                
                msg = {
                    "camera_name": CONFIG["camera_name"],
                    "counter": counter,
                    "time": last_frame_time
                }
                
                # Send (non-blocking với PUB-SUB)
                try:
                    sender.send_image(json.dumps(msg), frame)
                except zmq.Again:
                    print("[WARNING] Server timeout.")
                except Exception as e:
                    print(f"[ERROR] Send: {e}")
                
                del frame, counter, timestamp, data
            else:
                # Watchdog: kiểm tra timeout
                if time.time() - last_frame_time > watchdog_timeout:
                    print(f"[WARNING] No frame for {watchdog_timeout}s!")
                    last_frame_time = time.time()
                    # Kiểm tra inference process còn sống không
                    if not inference_proc.is_alive():
                        print("[ERROR] Inference process died! Restarting...")
                        inference_proc = Process(
                            target=inference_process,
                            args=(
                                CONFIG["model_name"],
                                model_config,
                                frame_queue,
                                result_queue,
                                scanning_event,
                                stop_event
                            )
                        )
                        inference_proc.start()
                    else:
                        # Chẩn đoán nguyên nhân
                        cam_lag = time.time() - cam_heartbeat.value
                        if cam_lag > watchdog_timeout:
                            print(f"[ERROR] Camera thread bị treo! (Không chụp ảnh trong {cam_lag:.1f}s)")
                        else:
                            print(f"[WARNING] Camera vẫn chạy (lag {cam_lag:.1f}s) -> Có thể Inference bị treo hoặc quá chậm.")
    
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...")
    
    finally:
        # 1. Ngắt kết nối MQTT
        if 'mqtt_client' in locals():
            mqtt_client.loop_stop()
            
        # 2. Báo hiệu dừng cho tất cả luồng/process
        stop_event.set()
        
        # 3. Đóng ZMQ Sender để ngắt các lệnh blocking (nếu có)
        if 'sender' in locals():
            try:
                sender.close()
            except:
                pass

        # 4. QUAN TRỌNG: Xả sạch hàng đợi Multiprocessing để tránh Deadlock khi join()
        # Nếu queue còn đầy, process con sẽ bị treo khi cố gắng flush dữ liệu lúc thoát
        print("[INFO] Draining queues...")
        while not result_queue.empty():
            try: result_queue.get_nowait()
            except: break
        while not frame_queue.empty():
            try: frame_queue.get_nowait()
            except: break

        # 5. Dừng Process Inference
        print("[INFO] Joining inference process...")
        inference_proc.join(timeout=3)
        if inference_proc.is_alive():
            print("[WARNING] Inference process hung, forcing terminate...")
            inference_proc.terminate()
            
        # 6. Dừng Camera Thread và giải phóng Camera
        # Cần join thread trước khi stop camera để tránh xung đột tài nguyên
        if 'camera_thread' in locals() and camera_thread.is_alive():
            camera_thread.join(timeout=2)
            # Nếu thread vẫn còn sống (kẹt trong capture), thoát ngay để tránh deadlock tại picam2.stop()
            if camera_thread.is_alive():
                print("[WARNING] Camera thread stuck. Force exiting...")
                os._exit(0)
            
        print("[INFO] Stopping camera...")
        try:
            picam2.stop()
        except:
            pass
        print("[INFO] Stopped.")
        os._exit(0)