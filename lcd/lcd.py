from RPLCD.i2c import CharLCD
import time

lcd = CharLCD('PCF8574', 0x27)  # đổi 0x27 nếu địa chỉ khác
lcd.write_string("Xin chao, Un!")
time.sleep(3)
lcd.clear()
lcd.write_string("Zero 2 W da OK!")
