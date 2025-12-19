import cv2
import imagezmq
import zmq
import json
import time
import threading
import random
import paho.mqtt.client as mqtt

# Config MQTT
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC_PUB = "pbl6/products"

CURRENT_CAM_NAME = None
IS_RUNNING = True

PRODUCTS = {
    "Sting": 10000,
    "Coca": 10000,
    "Pepsi": 10000,
    "Water": 5000,
    "Snack": 8000
}

def mqtt_publisher(client):
    print(f"[MQTT] Product Publisher started. Topic: {MQTT_TOPIC_PUB}")

    while True:
        time.sleep(10)
        label = random.choice(list(PRODUCTS.keys()))
        price = PRODUCTS[label]
        quantity = random.randint(1, 5)
        
        payload = {
            "label": label,
            "price": price,
            "quantity": quantity
        }
        try:
            client.publish(MQTT_TOPIC_PUB, json.dumps(payload))
            # print(f"[MQTT] Sent: {payload}")
        except Exception as e:
            print(f"[MQTT] Publish error: {e}")

def console_worker(client):
    global IS_RUNNING, CURRENT_CAM_NAME
    print("\n[INFO] Console Ready. Commands: 's' (SCAN), 'x' (STOP), 'q' (QUIT)\n")
    
    while IS_RUNNING:
        try:
            cmd = input().strip().lower()
            if cmd == 'q':
                IS_RUNNING = False
                break
            
            if not CURRENT_CAM_NAME:
                print("[WARNING] No camera connected yet.")
                continue
            
            topic = f"cmd/{CURRENT_CAM_NAME}"
            if cmd == 's':
                client.publish(topic, "SCAN")
                print(f"[CMD] Sent SCAN to {topic}")
            elif cmd == 'x':
                client.publish(topic, "STOP")
                print(f"[CMD] Sent STOP to {topic}")
        except Exception:
            pass

def main():
    """
    Server nhận video stream từ client và gửi dữ liệu giả lập qua MQTT mỗi 5s.
    """
    global CURRENT_CAM_NAME, IS_RUNNING

    # Khởi tạo ImageHub
    image_hub = imagezmq.ImageHub()

    # Set timeout để không block nếu không có frame
    image_hub.zmq_socket.setsockopt(zmq.RCVTIMEO, 100)

    print("[INFO] Server đang chạy. Đang chờ client...")

    # MQTT Client setup
    mqtt_client = mqtt.Client()
    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start()
    except Exception as e:
        print(f"[MQTT] Connection failed: {e}")

    # Start Threads
    threading.Thread(target=mqtt_publisher, args=(mqtt_client,), daemon=True).start()
    threading.Thread(target=console_worker, args=(mqtt_client,), daemon=True).start()

    connected_clients = set()
    display_size = (640, 640)  # Kích thước hiển thị
    count = 0
    try:
        while IS_RUNNING:
            try:
                msg, frame = image_hub.recv_image()
            except zmq.Again:
                # Không có frame → vẫn cho UI sống
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            
            # Parse JSON message
            counter = {}
            try:
                data = json.loads(msg)
                cam_name = data.get("camera_name", "Unknown")
                counter = data.get("counter", {})
            except Exception as e:
                # Chỉ gán cam_name = msg nếu lỗi xảy ra do msg không phải là JSON (giao thức cũ)
                # Nếu lỗi do print (sau khi đã parse JSON thành công), không được gán lại msg (vì msg là chuỗi JSON dài)
                if 'data' not in locals():
                    cam_name = msg

            CURRENT_CAM_NAME = cam_name

            if cam_name not in connected_clients:
                print(f"[INFO] Client connected: {cam_name}")
                connected_clients.add(cam_name)

            # Resize frame cho dễ nhìn
            display_frame = cv2.resize(
                frame,
                display_size,
                interpolation=cv2.INTER_NEAREST
            )

            # Hiển thị theo tên camera
            cv2.imshow(cam_name, display_frame)

            # Phím q để thoát
            if cv2.waitKey(1) & 0xFF == ord('q'):
                IS_RUNNING = False
                break

            # Gửi ACK cho client
            image_hub.send_reply(b'OK')
            if count % 10 == 0:
                print(f"[{cam_name}] Counter: {counter}")
            count += 1

    except (KeyboardInterrupt, SystemExit):
        print("\n[INFO] Stopping server...")

    finally:
        cv2.destroyAllWindows()
        print("[INFO] Server stopped.")

if __name__ == "__main__":
    main()
