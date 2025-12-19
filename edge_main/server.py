import cv2
import imagezmq
import zmq
import json
import time
import threading
import random
import paho.mqtt.client as mqtt

# Config MQTT
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC_PUB = "pbl6/products"

PRODUCTS = {
    "Sting": 10000,
    "Coca": 10000,
    "Pepsi": 10000,
    "Water": 5000,
    "Snack": 8000
}

def mqtt_publisher():
    client = mqtt.Client()
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_start()
        print(f"[MQTT] Publisher started. Topic: {MQTT_TOPIC_PUB}")
    except Exception as e:
        print(f"[MQTT] Connection failed: {e}")
        return

    while True:
        time.sleep(5)
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
            print(f"[MQTT] Sent: {payload}")
        except Exception as e:
            print(f"[MQTT] Publish error: {e}")

def main():
    """
    Server nhận video stream từ client và gửi dữ liệu giả lập qua MQTT mỗi 5s.
    """

    # Khởi tạo ImageHub
    image_hub = imagezmq.ImageHub()

    # Set timeout để không block nếu không có frame
    image_hub.zmq_socket.setsockopt(zmq.RCVTIMEO, 100)

    print("[INFO] Server đang chạy. Đang chờ client...")

    # Start MQTT Publisher Thread
    threading.Thread(target=mqtt_publisher, daemon=True).start()

    connected_clients = set()
    display_size = (640, 640)  # Kích thước hiển thị

    try:
        while True:
            try:
                msg, frame = image_hub.recv_image()
            except zmq.Again:
                # Không có frame → vẫn cho UI sống
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            
            # Parse JSON message
            try:
                data = json.loads(msg)
                cam_name = data.get("camera_name", "Unknown")
                counter = data.get("counter", {})
                # print(f"[{cam_name}] Counter: {counter}") 
            except Exception:
                cam_name = msg

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
                break

            # Gửi ACK cho client
            image_hub.send_reply(b'OK')

    except (KeyboardInterrupt, SystemExit):
        print("\n[INFO] Stopping server...")

    finally:
        cv2.destroyAllWindows()
        print("[INFO] Server stopped.")

if __name__ == "__main__":
    main()
