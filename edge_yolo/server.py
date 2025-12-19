import cv2
import json
import imagezmq
import traceback
import paho.mqtt.client as mqtt
import threading
import sys
import zmq

# Cấu hình MQTT
MQTT_BROKER = "localhost"  # Địa chỉ IP của Broker (thường là máy chạy server này)
MQTT_PORT = 1883

# Biến toàn cục để chia sẻ trạng thái giữa luồng console và luồng chính
CURRENT_CAM_NAME = None
IS_RUNNING = True

def console_worker(mqtt_client):
    """Luồng xử lý nhập lệnh từ console."""
    global CURRENT_CAM_NAME, IS_RUNNING
    print("\n[INFO] Console Control Ready.")
    print("Commands: 's' (SCAN), 'x' (STOP), 'p [amount]' (PAY), 'q' (QUIT)\n")
    
    while IS_RUNNING:
        try:
            # input() sẽ block luồng này, nhưng không ảnh hưởng luồng video chính
            cmd_str = input()
            if not cmd_str: continue
            
            parts = cmd_str.strip().split()
            cmd = parts[0].lower()
            
            if cmd == 'q':
                IS_RUNNING = False
                break
            
            if not CURRENT_CAM_NAME:
                print("[WARNING] Chưa có camera kết nối.")
                continue
                
            topic = f"cmd/{CURRENT_CAM_NAME}"
            
            if cmd == 's':
                mqtt_client.publish(topic, "SCAN")
                print(f"[CMD] Sent SCAN to {topic}")
            elif cmd == 'x':
                mqtt_client.publish(topic, "STOP")
                print(f"[CMD] Sent STOP to {topic}")
            elif cmd == 'p':
                amount = parts[1] if len(parts) > 1 else "150000"
                mqtt_client.publish(topic, f"TONGTIEN: {amount}")
                print(f"[CMD] Sent PAYMENT: {amount} to {topic}")
            else:
                print(f"[INFO] Lệnh không hợp lệ: {cmd}")
        except Exception as e:
            print(f"[ERROR] Console: {e}")

def main():
    """
    Khởi chạy server để nhận và hiển thị video stream từ các client.
    Đồng thời đóng vai trò Controller gửi lệnh MQTT (SCAN/STOP/PAYMENT).
    """
    global CURRENT_CAM_NAME, IS_RUNNING

    # 1. Khởi tạo MQTT Client
    mqtt_client = mqtt.Client()
    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start()
        print(f"[INFO] MQTT Connected to {MQTT_BROKER}")
    except Exception as e:
        print(f"[WARNING] Không thể kết nối MQTT: {e}. Chức năng điều khiển sẽ không hoạt động.")

    # Khởi tạo ImageHub để lắng nghe kết nối từ các client.
    image_hub = imagezmq.ImageHub()
    # [QUAN TRỌNG] Set timeout 100ms để không bị treo nếu Client ngừng gửi
    image_hub.zmq_socket.setsockopt(zmq.RCVTIMEO, 100)

    print("[INFO] Server đang chạy. Đang chờ kết nối từ client...")
    connected_clients = set()
    display_size = (640, 640)  # Kích thước cửa sổ hiển thị mong muốn
    
    # Khởi chạy luồng console
    t = threading.Thread(target=console_worker, args=(mqtt_client,), daemon=True)
    t.start()

    try:
        # Vòng lặp vô tận để nhận và hiển thị các khung hình
        while IS_RUNNING:
            # Nhận tên camera và khung hình từ client
            try:
                msg, frame = image_hub.recv_image()
            except zmq.Again:
                # Không nhận được ảnh (Timeout), tiếp tục vòng lặp để UI không bị treo
                cv2.waitKey(1)
                continue

            try:
                data = json.loads(msg)
                cam_name = data.get("camera_name", "Unknown")
                current = data.get("current", {})
                total = data.get("total", {})
                
                CURRENT_CAM_NAME = cam_name
            except (json.JSONDecodeError, TypeError):
                cam_name = msg

            # In thông báo nếu đây là client mới
            if cam_name not in connected_clients:
                print(f"[INFO] Nhan duoc ket noi moi tu client: {cam_name}")
                connected_clients.add(cam_name)

            # Thay đổi kích thước frame để hiển thị lớn hơn
            display_frame = cv2.resize(frame, display_size, interpolation=cv2.INTER_NEAREST)
            
            # Hiển thị hướng dẫn điều khiển
            cv2.putText(display_frame, "Console: s(Scan) x(Stop) p(Pay)", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            

            # Hiển thị khung hình trong một cửa sổ có tên là tên của camera
            cv2.imshow(cam_name, display_frame)

            # Chờ 1ms và kiểm tra nếu người dùng nhấn phím 'q' để thoát
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                IS_RUNNING = False
                break

            # Gửi tín hiệu 'OK' về cho client để xác nhận đã nhận ảnh
            image_hub.send_reply(b'OK')

    except (KeyboardInterrupt, SystemExit):
        print("\n[INFO] Đang dừng server...")
    finally:
        mqtt_client.loop_stop()
        # Dọn dẹp, đóng tất cả các cửa sổ OpenCV
        cv2.destroyAllWindows()
        print("[INFO] Server đã dừng.")

if __name__ == '__main__':
    main()