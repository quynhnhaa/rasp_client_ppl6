# from RPLCD.i2c import CharLCD
# import time

# # Khởi tạo LCD (nếu i2cdetect ra 0x27 hay 0x3f thì sửa lại cho đúng)
# lcd = CharLCD('PCF8574', 0x27)


# lcd.clear()


# lcd.cursor_pos = (0, 0)
# lcd.write_string("Xin chao, Un!")

# # Dòng 2
# lcd.cursor_pos = (1, 0)
# lcd.write_string("Zero 2W da OK!")

# time.sleep(5)
# lcd.clear()

from RPLCD.i2c import CharLCD
import time

lcd = CharLCD('PCF8574', 0x27)  # đổi địa chỉ nếu cần, VD 0x3f
lcd.clear()

text = "Un rat xinh dep nhung LCD ngan qua!"  # ví dụ chuỗi dài

# Thêm khoảng trống để tạo hiệu ứng trượt mượt
text = " " * 16 + text + " " * 16

while True:
    for i in range(len(text) - 15):
        lcd.cursor_pos = (0, 0)
        lcd.write_string(text[i:i+16])  # hiển thị khung 16 ký tự
        time.sleep(0.3)
