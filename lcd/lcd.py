from RPLCD.i2c import CharLCD
import time

# Khởi tạo LCD (nếu i2cdetect ra 0x27 hay 0x3f thì sửa lại cho đúng)
lcd = CharLCD('PCF8574', 0x27)

# Xóa màn hình
lcd.clear()

# Dòng 1
lcd.cursor_pos = (0, 0)
lcd.write_string("Xin chao, Un!")

# Dòng 2
lcd.cursor_pos = (1, 0)
lcd.write_string("Zero 2W da OK!")

time.sleep(5)
lcd.clear()
