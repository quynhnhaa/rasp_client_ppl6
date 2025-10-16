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
from time import sleep

def long_string(display, text='', num_line=1, num_cols=16):
    """ 
    Hiển thị chuỗi dài bằng cách cuộn từ phải sang trái
    Parameters:
        display: đối tượng LCD (VD: lcd)
        text: chuỗi muốn hiển thị
        num_line: dòng (1 hoặc 2)
        num_cols: số cột (thường là 16)
    """
    if len(text) <= num_cols:
        # Nếu chuỗi ngắn, in thẳng
        display.cursor_pos = (num_line - 1, 0)
        display.write_string(text.ljust(num_cols))
        return

    # Thêm khoảng trắng để hiệu ứng mượt hơn
    text = ' ' * num_cols + text + ' ' * num_cols

    for i in range(len(text) - num_cols + 1):
        display.cursor_pos = (num_line - 1, 0)
        display.write_string(text[i:i + num_cols])
        sleep(0.3)


# --- Main ---
lcd = CharLCD('PCF8574', 0x27)
lcd.clear()

# Dòng 1 (tĩnh)
lcd.cursor_pos = (0, 0)
lcd.write_string("Xin chao, Un!")

# Dòng 2 (chạy chữ từ phải sang trái)
long_string(lcd, "Zero 2W da OK! Chao mung Un nhe!", num_line=2)

sleep(1)
lcd.clear()
lcd.close()