import cv2
import json
import imagezmq
import traceback
import paho.mqtt.client as mqtt
import zmq

# Cấu hình MQTT
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC_SUB = "scan/#"

CAMERA_DATA = {} # Lưu dữ liệu từ MQTT: {"cam_name": {"current": {}, "total": {}, "money": 0}}

def on_mqtt_connect(client, userdata, flags, rc):
    print(f"[MQTT] Connected with result code {rc}")
    client.subscribe(MQTT_TOPIC_SUB)

def on_mqtt_message(client, userdata, msg):
    global CAMERA_DATA
    try:
        # Topic: scan/camera_name
        parts = msg.topic.split('/')
        if len(parts) > 1:
            cam_name = parts[1]
            payload = json.loads(msg.payload.decode())
            CAMERA_DATA[cam_name] = payload
    except Exception as e:
        print(f"[MQTT] Error: {e}")

def main():
    """
    Khởi chạy server để nhận và hiển thị video stream từ các client.
    """
    # Khởi tạo ImageHub để lắng nghe kết nối từ các client.
    image_hub = imagezmq.ImageHub()
    # Set timeout để không bị treo nếu không có ảnh (100ms)
    image_hub.zmq_socket.setsockopt(zmq.RCVTIMEO, 100)

    # Khởi tạo MQTT Client
    mqtt_client = mqtt.Client()
    mqtt_client.on_connect = on_mqtt_connect
    mqtt_client.on_message = on_mqtt_message
    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start()
    except Exception as e:
        print(f"[ERROR] MQTT Connection: {e}")

    print("[INFO] Server đang chạy. Đang chờ kết nối từ client...")
    connected_clients = set()
    display_size = (640, 640)  # Kích thước cửa sổ hiển thị mong muốn

    try:
        # Vòng lặp vô tận để nhận và hiển thị các khung hình
        while True:
            # Nhận tên camera và khung hình từ client
            try:
                cam_name, frame = image_hub.recv_image()
            except zmq.Again:
                # Timeout, kiểm tra phím bấm để không treo UI
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # In thông báo nếu đây là client mới
            if cam_name not in connected_clients:
                print(f"[INFO] Nhan duoc ket noi moi tu client: {cam_name}")
                connected_clients.add(cam_name)

            # Thay đổi kích thước frame để hiển thị lớn hơn
            display_frame = cv2.resize(frame, display_size, interpolation=cv2.INTER_NEAREST)

            # Lấy dữ liệu từ MQTT để vẽ lên frame
            data = CAMERA_DATA.get(cam_name, {})
            current = data.get("current", {})
            total = data.get("total", {})
            money = data.get("money", 0)

            # Vẽ thông tin
            cv2.putText(display_frame, f"MONEY: {money:,} VND", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            y = 60
            if current:
                cv2.putText(display_frame, "SCANNING:", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                y += 25
                for k, v in current.items():
                    cv2.putText(display_frame, f"- {k}: {v}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    y += 20
            
            y += 10
            cv2.putText(display_frame, "TOTAL:", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y += 25
            for k, v in total.items():
                cv2.putText(display_frame, f"- {k}: {v}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                y += 20

            # Hiển thị khung hình trong một cửa sổ có tên là tên của camera
            cv2.imshow(cam_name, display_frame)

            # Chờ 1ms và kiểm tra nếu người dùng nhấn phím 'q' để thoát
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Gửi tín hiệu 'OK' về cho client để xác nhận đã nhận ảnh
            image_hub.send_reply(b'OK')

    except (KeyboardInterrupt, SystemExit):
        print("\n[INFO] Đang dừng server...")
    finally:
        # Dọn dẹp, đóng tất cả các cửa sổ OpenCV
        cv2.destroyAllWindows()
        print("[INFO] Server đã dừng.")

if __name__ == '__main__':
    main()