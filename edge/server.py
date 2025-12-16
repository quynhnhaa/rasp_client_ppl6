import cv2
import imagezmq
import traceback

def main():
    """
    Khởi chạy server để nhận và hiển thị video stream từ các client.
    """
    # Khởi tạo ImageHub để lắng nghe kết nối từ các client.
    # Mặc định, nó sẽ lắng nghe trên tất cả các IP của máy ở port 5555.
    image_hub = imagezmq.ImageHub()

    print("[INFO] Server đang chạy. Đang chờ kết nối từ client...")

    try:
        # Vòng lặp vô tận để nhận và hiển thị các khung hình
        while True:
            # Nhận tên camera và khung hình từ client
            # Lệnh này sẽ block cho đến khi có ảnh mới
            cam_name, frame = image_hub.recv_image()

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