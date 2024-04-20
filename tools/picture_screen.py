import pyautogui as pag
import pywinctl as pwc


def get_screen():
	window = pwc.getWindowsWithTitle("ムービー収録")[0] # QuickTime Player
	x, y = window.topleft
	width, height = window.size
	sc = pag.screenshot(region=(x, y, width, height))
	return sc
