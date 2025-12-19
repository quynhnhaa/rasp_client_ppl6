import cv2
import imagezmq
import zmq

def main():
    """
    Server chỉ nhận và hiển thị video stream từ client (KHÔNG MQTT).
    """

    # Khởi tạo ImageHub
    image_hub = imagezmq.ImageHub()

    # Set timeout để không block nếu không có frame
    image_hub.zmq_socket.setsockopt(zmq.RCVTIMEO, 100)

    print("[INFO] Server đang chạy (IMAGE ONLY). Đang chờ client...")

    connected_clients = set()
    display_size = (640, 640)  # Kích thước hiển thị

    try:
        while True:
            try:
                cam_name, frame = image_hub.recv_image()
            except zmq.Again:
                # Không có frame → vẫn cho UI sống
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

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
