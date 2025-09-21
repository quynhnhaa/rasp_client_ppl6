# import socket
# import time
# import cv2
# import imagezmq
# from picamera2 import Picamera2

# server_ip = "192.168.1.10"  
# sender = imagezmq.ImageSender(connect_to=f"tcp://{server_ip}:5555")

# rpi_name = socket.gethostname()
# picam2 = Picamera2()
# config = picam2.create_preview_configuration({"size": (640, 480)})
# picam2.configure(config)
# picam2.start()

# while True:
#     frame = picam2.capture_array()
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     sender.send_image(rpi_name, frame)
#     time.sleep(0.05)


import socket
import time
import cv2
import imagezmq
from picamera2 import Picamera2

server_ip = "172.20.10.12"
sender = imagezmq.ImageSender(connect_to=f"tcp://{server_ip}:5555")

rpi_name = socket.gethostname()
picam2 = Picamera2()
config = picam2.create_preview_configuration({"size": (640, 480)})
picam2.configure(config)
picam2.start()

while True:
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    ret, jpg_buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
    sender.send_jpg(rpi_name, jpg_buffer)

    if cv2.waitKey(1) == ord('q'):
        break
