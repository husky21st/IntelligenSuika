import numpy as np
import cv2
from collections import OrderedDict
from time import perf_counter
from screen_operation import get_screen, active_window
import json
from matplotlib import pyplot as plt
from utils import rotate_img, detect_xy


FRUITS_ROTATION = dict((
	(1, 18),
	(2, 12),
	(3, 6),
	(4, 12),
	(5, 9),
	(6, 6),
	(7, 9),
	(8, 15),
	(9, 9),
	(10, 6),
	(11, 6),
))


FRUITS_THRESHOLD = dict((
	(1, 0.970),
	(2, 0.968),
	(3, 0.970),
	(4, 0.990),
	(5, 0.985),
	(6, 0.980),
	(7, 0.994),
	(8, 0.996),
	(9, 0.990),
	(10, 0.973),
	(11, 0.935),
))

FRUITS_TEMPLATE = OrderedDict({i: cv2.imread(f"./../assets/f{str(i)}.png", cv2.IMREAD_UNCHANGED) for i in range(11, 0, -1)})
FRUITS_TEMPLATE_VAR_ANGLE_IPAD = OrderedDict({label: [rotate_img(fruit, angle, 1.0) for angle in range(0, 360, 360//FRUITS_ROTATION[label])] for label, fruit in FRUITS_TEMPLATE.items()})
FRUITS_TEMPLATE_NEXT_IPAD = OrderedDict({label: [rotate_img(fruit, -45, 0.8)] for label, fruit in FRUITS_TEMPLATE.items()})


def get_state_from_ipad():
	# sc = get_screen()
	# sc = cv2.cvtColor(np.array(sc, dtype=np.uint8), cv2.COLOR_RGBA2BGR)
	sct1345678 = cv2.imread('./../tmp/001.png')
	sct1234567 = cv2.imread('./../tmp/004.png')
	sct12468910 = cv2.imread('./../tmp/006.png')
	sct1234567811 = cv2.imread('./../tmp/008.png')
	sct = cv2.imread('./../tmp/0f5.png')
	sc = sct1345678
	# field = sc[335:986, 107:625, :]
	field = sc[409:986, 107:625, :]  # delete overlap
	haved_area = sc[267:365, 107:625, :]
	next_area = sc[70:135, 408:450, :]

	# field
	field_fruits = list()
	for label in range(11, 0, -1):
		field, fruit_locations = detect_xy(
			FRUITS_TEMPLATE_VAR_ANGLE_IPAD[label],
			field,
			threshold=FRUITS_THRESHOLD[label],
			nms_thresh=0.2)

		if not fruit_locations:
			continue
		# field_fruits.append(**fruit_locations, label)

	return sc


if __name__ == '__main__':
	# active_window()
	state = get_state_from_ipad()
