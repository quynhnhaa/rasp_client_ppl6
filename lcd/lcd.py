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
    Hiển thị chuỗi dài theo kiểu:
    - Nếu chuỗi ngắn hơn num_cols → in thẳng
    - Nếu chuỗi dài hơn num_cols → in 16 ký tự đầu, dừng 1s, rồi cuộn từ PHẢI sang TRÁI
    """
    row = num_line - 1  # RPLCD dùng index bắt đầu từ 0

    if len(text) > num_cols:
        # In 16 ký tự đầu tiên trước
        display.cursor_pos = (row, 0)
        display.write_string(text[:num_cols].ljust(num_cols))
        sleep(1)

        # Thêm khoảng trắng để cuộn mượt
        scroll_text = text + ' ' * num_cols

        # Cuộn từ phải sang trái
        for i in range(len(scroll_text) - num_cols + 1):
            display.cursor_pos = (row, 0)
            display.write_string(scroll_text[i:i + num_cols])
            sleep(0.2)

        sleep(1)
    else:
        # Chuỗi ngắn, in thẳng
        display.cursor_pos = (row, 0)
        display.write_string(text.ljust(num_cols))


# --- Main ---
lcd = CharLCD('PCF8574', 0x27)
lcd.clear()

# Dòng 1 tĩnh
lcd.cursor_pos = (0, 0)
lcd.write_string("Xin chao, Un!")

# Dòng 2: chạy chữ từ phải sang trái (theo logic gốc)
long_string(lcd, "Zero 2W da OK! Chao mung Un nhe!", num_line=2)

sleep(1)
lcd.close(clear=True)