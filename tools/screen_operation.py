import pyautogui as pag
import pywinctl as pwc
import time
import serial


def get_screen():
	"""QuickTime Player画面のスクリーンショットを取得

	"""
	window = pwc.getWindowsWithTitle("ムービー収録")[0]  # QuickTime Player
	x, y = window.topleft
	width, height = window.size
	sc = pag.screenshot(region=(x, y, width, height))
	return sc


def click_screen(ser, x: float):
	"""実機上で指定の場所まで移動してからクリックする

	:param ser: 接続済みのシリアル通信
	:param x: クリックする場所 -1~1の範囲をとる
	:type x: float
	"""
	assert -1 <= x <= 1
	ser.write(bytes(str(round(x, 4)), 'utf-8'))
	time.sleep(2)


if __name__ == '__main__':
	port = "/dev/cu.usbserial-0001"
	port_rate = 115200
	ser = serial.Serial(port, port_rate)
	time.sleep(5)
	click_screen(ser, 1)
	click_screen(ser, -0.99999)
	ser.close()
