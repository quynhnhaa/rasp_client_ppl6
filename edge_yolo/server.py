import cv2
import json
import imagezmq
import traceback
import paho.mqtt.client as mqtt

# Cấu hình MQTT
MQTT_BROKER = "localhost"  # Địa chỉ IP của Broker (thường là máy chạy server này)
MQTT_PORT = 1883

def main():
    """
    Khởi chạy server để nhận và hiển thị video stream từ các client.
    Đồng thời đóng vai trò Controller gửi lệnh MQTT (SCAN/STOP/PAYMENT).
    """
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

    print("[INFO] Server đang chạy. Đang chờ kết nối từ client...")
    connected_clients = set()
    display_size = (640, 640)  # Kích thước cửa sổ hiển thị mong muốn
    
    # Biến lưu tên camera hiện tại để gửi lệnh điều khiển
    current_cam_name = None

    try:
        # Vòng lặp vô tận để nhận và hiển thị các khung hình
        while True:
            # Nhận tên camera và khung hình từ client
            # Lệnh này sẽ block cho đến khi có ảnh mới
            msg, frame = image_hub.recv_image()

            try:
                data = json.loads(msg)
                cam_name = data.get("camera_name", "Unknown")
                current = data.get("current", {})
                total = data.get("total", {})
                
                current_cam_name = cam_name
            except (json.JSONDecodeError, TypeError):
                cam_name = msg

            # In thông báo nếu đây là client mới
            if cam_name not in connected_clients:
                print(f"[INFO] Nhan duoc ket noi moi tu client: {cam_name}")
                connected_clients.add(cam_name)

            # Thay đổi kích thước frame để hiển thị lớn hơn
            display_frame = cv2.resize(frame, display_size, interpolation=cv2.INTER_NEAREST)
            
            # Hiển thị hướng dẫn điều khiển
            cv2.putText(display_frame, "Controls: [S]can [X]Stop [P]ay", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            

            # Hiển thị khung hình trong một cửa sổ có tên là tên của camera
            cv2.imshow(cam_name, display_frame)

            # Chờ 1ms và kiểm tra nếu người dùng nhấn phím 'q' để thoát
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and current_cam_name:
                topic = f"cmd/{current_cam_name}"
                mqtt_client.publish(topic, "SCAN")
                print(f"[CMD] Sent SCAN to {topic}")
            elif key == ord('x') and current_cam_name:
                topic = f"cmd/{current_cam_name}"
                mqtt_client.publish(topic, "STOP")
                print(f"[CMD] Sent STOP to {topic}")
            elif key == ord('p') and current_cam_name:
                topic = f"cmd/{current_cam_name}"
                # Giả lập tính tổng tiền (bạn có thể thay đổi logic tính toán ở đây)
                mqtt_client.publish(topic, "TONGTIEN: 150000") 
                print(f"[CMD] Sent PAYMENT to {topic}")

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