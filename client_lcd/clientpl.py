"""
- Gửi ảnh JPEG chất lượng cao
- Giới hạn FPS để tiết kiệm tài nguyên
- Tự động reconnect nếu mất kết nối
- Nhận và hiển thị thông tin sản phẩm trên LCD qua MQTT
"""

import os
import time
import socket
import cv2
import numpy as np
import imagezmq
from picamera2 import Picamera2
import json

try:
    from RPLCD.i2c import CharLCD
    RPLCD_AVAILABLE = True
except ImportError:
    RPLCD_AVAILABLE = False
    print("[WARNING] RPLCD library not found. LCD functionality will be disabled.")

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    print("[WARNING] paho-mqtt library not found. MQTT functionality will be disabled.")

# ========== Cấu hình hệ thống ==========
CONFIG = {
    "server_ip": os.getenv("server_ip", "172.20.10.12"),
    "port": int(os.getenv("port", "5555")),
    "jpeg_quality": int(os.getenv("jpeg_quality", "90")),
    "target_fps": int(os.getenv("target_fps", "20")),
    "frame_size":  list(map(int, os.getenv("frame_size", "640x640").split("x")))
}

CAMERA_NAME = socket.gethostname()
CONNECT_TO = f"tcp://{CONFIG['server_ip']}:{CONFIG['port']}"

# ========== Cấu hình MQTT ==========
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "pbl6/products"

# ========== Khởi tạo LCD ==========
def init_lcd():
    if not RPLCD_AVAILABLE:
        return None
    try:
        # Thử các địa chỉ I2C phổ biến
        for address in [0x27, 0x3f]:
            try:
                lcd = CharLCD('PCF8574', address)
                lcd.clear()
                lcd.write_string("Waiting for...")
                print(f"[INFO] LCD initialized at address {hex(address)}.")
                return lcd
            except Exception:
                continue
        raise IOError("Could not find LCD at common I2C addresses.")
    except Exception as e:
        print(f"[ERROR] Could not initialize LCD: {e}")
        return None

# ========== Hiển thị trên LCD ==========
def display_on_lcd(lcd, product, price):
    if lcd is None:
        return
    try:
        lcd.clear()
        # Dòng 1: Tên sản phẩm
        lcd.cursor_pos = (0, 0)
        lcd.write_string(str(product)[:16])
        # Dòng 2: Giá tiền
        lcd.cursor_pos = (1, 0)
        price_str = f"Total: {price:,.0f} VND"
        lcd.write_string(price_str[:16])
    except Exception as e:
        print(f"[ERROR] Could not write to LCD: {e}")

# ========== MQTT Callbacks ==========
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"[MQTT] Connected to broker. Subscribing to {MQTT_TOPIC}")
        client.subscribe(MQTT_TOPIC)
    else:
        print(f"[MQTT] Failed to connect, return code {rc}")

def on_message(client, userdata, msg):
    lcd = userdata.get('lcd')
    print(f"[MQTT] Received message on topic {msg.topic}")
    try:
        data = json.loads(msg.payload.decode())
        product = data.get("product", "N/A")
        price = data.get("price", 0)
        display_on_lcd(lcd, product, price)
    except json.JSONDecodeError:
        print("[WARNING] Received invalid JSON from MQTT.")
    except Exception as e:
        print(f"[ERROR] Error processing MQTT message: {e}")

# ========== Khởi tạo camera ==========
def init_camera():
    cam = Picamera2()
    cam_config = cam.create_preview_configuration(main={"size": CONFIG["frame_size"], "format": "RGB888"})
    cam.configure(cam_config)
    cam.start()
    return cam

# ========== Gửi ảnh qua ImageZMQ ==========
def send_frame(sender, frame, quality):
    cam_id = f"{CAMERA_NAME}"
    ok, jpg_buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if ok:
        sender.send_jpg(cam_id, jpg_buffer.tobytes())

# ========== Vòng lặp chính ==========
def main():
    print(f"[INFO] Connecting to image server at {CONNECT_TO}")
    sender = None
    backoff = 1.0

    cam = init_camera()
    lcd = init_lcd()
    
    # Setup MQTT Client
    mqtt_client = None
    if MQTT_AVAILABLE:
        userdata = {'lcd': lcd}
        mqtt_client = mqtt.Client()
        mqtt_client.user_data_set(userdata)
        mqtt_client.on_connect = on_connect
        mqtt_client.on_message = on_message
        try:
            mqtt_client.connect_async(MQTT_BROKER, MQTT_PORT, 60)
            mqtt_client.loop_start() # Start non-blocking loop
        except Exception as e:
            print(f"[ERROR] Could not start MQTT client: {e}")
            mqtt_client = None # Disable MQTT if connection fails
    
    last_frame_time = time.time()
    frame_interval = 1.0 / CONFIG["target_fps"]

    try:
        while True:
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
                    print("[INFO] Connected to image server.")
                    backoff = 1.0
                except Exception as e:
                    print(f"[ERROR] Cannot connect to image server: {e}. Retrying in {backoff:.1f}s")
                    time.sleep(backoff)
                    backoff = min(10.0, backoff * 1.5)
                    continue

            try:
                send_frame(sender, frame, quality)
            except Exception as e:
                print(f"[ERROR] Failed to send frame: {e}")
                if sender:
                    sender.close()
                sender = None
                time.sleep(backoff)
                backoff = min(10.0, backoff * 1.5)

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user. Exiting...")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
    finally:
        if sender:
            sender.close()
        if mqtt_client:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
        if lcd:
            lcd.close(clear=True)
        cam.stop()
        print("[INFO] Cleanup complete. Exiting.")

if __name__ == "__main__":
    main()