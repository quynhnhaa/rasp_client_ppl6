# # server.py
# import imagezmq
# import cv2

# # Khởi tạo ImageHub để nhận ảnh
# imageHub = imagezmq.ImageHub()

# while True:
#     rpiName, frame = imageHub.recv_image()
#     imageHub.send_reply(b'OK')  # gửi tín hiệu xác nhận

#     # Hiển thị frame nhận được
#     cv2.imshow(f"Live from {rpiName}", frame)
#     if cv2.waitKey(1) == ord('q'):
#         break

# cv2.destroyAllWindows()

import imagezmq
import cv2
import numpy as np  # cần import numpy

imageHub = imagezmq.ImageHub()

while True:
    rpiName, jpg_buffer = imageHub.recv_jpg()
    imageHub.send_reply(b'OK')

    np_array = np.frombuffer(jpg_buffer, dtype=np.uint8)
    frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    cv2.imshow(f"Live from {rpiName}", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()


