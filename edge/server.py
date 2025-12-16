import cv2
import imagezmq
import traceback
import numpy as np

def main():
    """
    Khởi chạy server để nhận và hiển thị video stream từ các client.
    """
    # Khởi tạo ImageHub để lắng nghe kết nối từ các client.
    # Mặc định, nó sẽ lắng nghe trên tất cả các IP của máy ở port 5555.
    image_hub = imagezmq.ImageHub()

    print("[INFO] Server đang chạy. Đang chờ kết nối từ client...")
    connected_clients = set()

    try:
        # Vòng lặp vô tận để nhận và hiển thị các khung hình
        while True:
            # Nhận tên camera và khung hình từ client
            # Lệnh này sẽ block cho đến khi có ảnh mới
            # 1. Nhận chuỗi byte JPEG từ client
            cam_name, jpg_buffer = image_hub.recv_jpg()
            # 2. Giải nén chuỗi byte thành ảnh OpenCV
            frame = cv2.imdecode(np.frombuffer(jpg_buffer, dtype='uint8'), cv2.IMREAD_COLOR)

            # In thông báo nếu đây là client mới
            if cam_name not in connected_clients:
                print(f"[INFO] Nhan duoc ket noi moi tu client: {cam_name}")
                connected_clients.add(cam_name)

            # Hiển thị khung hình trong một cửa sổ có tên là tên của camera
            cv2.imshow(cam_name, frame)

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