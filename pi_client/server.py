import cv2
import imagezmq
import numpy as np

def main():
    """
    Khởi chạy server để nhận và hiển thị video stream từ các client C++.
    """
    # Khởi tạo ImageHub. Nó sẽ tạo một socket REP và lắng nghe trên tất cả
    # các địa chỉ IP của máy chủ tại port 5555.
    # Port này phải khớp với SERVER_PORT trong file main.cpp
    image_hub = imagezmq.ImageHub()

    print("[INFO] Server đang chạy. Đang chờ kết nối từ client...")
    
    # Dictionary để theo dõi các client đã kết nối
    connected_clients = set()

    try:
        # Vòng lặp vô tận để nhận và hiển thị các khung hình
        while True:
            # 1. Nhận dữ liệu từ client (hàm này sẽ block cho đến khi có dữ liệu)
            # imagezmq sẽ tự động xử lý việc nhận message đa phần:
            # - Phần 1: Tên camera (dạng string)
            # - Phần 2: Dữ liệu ảnh (dạng bytes)
            cam_name, jpg_buffer = image_hub.recv_jpg()

            # In thông báo nếu đây là một client mới
            if cam_name not in connected_clients:
                print(f"[INFO] Nhận được kết nối mới từ client: {cam_name}")
                connected_clients.add(cam_name)

            # 2. Giải nén buffer JPEG thành ảnh OpenCV
            # Dùng np.frombuffer để chuyển byte thành numpy array, sau đó giải nén
            try:
                frame = cv2.imdecode(np.frombuffer(jpg_buffer, dtype='uint8'), cv2.IMREAD_COLOR)
                
                # Nếu giải nén thành công, hiển thị ảnh
                if frame is not None:
                    cv2.imshow(cam_name, frame)
                else:
                    print(f"[WARN] Không thể giải nén ảnh từ client: {cam_name}")

            except Exception as e:
                print(f"[ERROR] Lỗi khi xử lý ảnh từ {cam_name}: {e}")


            # 3. Gửi tín hiệu 'OK' về cho client để xác nhận đã nhận ảnh
            # Điều này là bắt buộc trong mô hình REQ/REP để client có thể gửi frame tiếp theo.
            image_hub.send_reply(b'OK')

            # Chờ 1ms và kiểm tra nếu người dùng nhấn phím 'q' để thoát
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except (KeyboardInterrupt, SystemExit):
        print("\n[INFO] Đang dừng server...")
    finally:
        # Dọn dẹp, đóng tất cả các cửa sổ OpenCV
        cv2.destroyAllWindows()
        # Đóng ImageHub để giải phóng tài nguyên
        image_hub.close()
        print("[INFO] Server đã dừng.")

if __name__ == '__main__':
    main()
