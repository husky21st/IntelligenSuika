import numpy as np
import cv2
from collections import OrderedDict
from time import perf_counter
from screen_operation import get_screen, active_window
import json
from matplotlib import pyplot as plt
from utils import *


MAX_FRUIT_NUM = 60

FRUITS_ROTATION = dict((
	(1, 18),
	(2, 12),
	(3, 9),
	(4, 12),
	(5, 9),
	(6, 9),
	(7, 9),
	(8, 20),
	(9, 9),
	(10, 6),
	(11, 6),
))


FRUITS_THRESHOLD = dict((
	(1, 0.970),
	(2, 0.968),
	(3, 0.950),
	(4, 0.990),
	(5, 0.985),
	(6, 0.970),
	(7, 0.994),
	(8, 0.995),
	(9, 0.994),
	(10, 0.973),
	(11, 0.935),
))

FRUITS_THRESHOLD_HIGH = dict((
	(1, 0.989),
	(2, 0.987),
	(3, 0.987),
	(4, 0.996),
	(5, 0.997),
))

FRUITS_TEMPLATE_IPAD = OrderedDict({label: cv2.imread(f"./../assets/ipad/f{str(label)}.png", cv2.IMREAD_UNCHANGED) for label in range(11, 0, -1)})
FRUITS_TEMPLATE_VAR_ANGLE_IPAD = OrderedDict({label: [rotate_img(fruit, angle, 1.0) for angle in range(0, 360, 360//FRUITS_ROTATION[label])] for label, fruit in FRUITS_TEMPLATE_IPAD.items()})
FRUITS_TEMPLATE_NEXT_IPAD = OrderedDict({label: [cv2.resize(rotate_img(fruit, -45, 1.0), dsize=None, fx=0.4, fy=0.4)] for label, fruit in FRUITS_TEMPLATE_IPAD.items() if label <= 5})

FRUITS_TEMPLATE_YOUTUBE = OrderedDict({label: cv2.imread(f"./../assets/youtube/f{str(label)}.png", cv2.IMREAD_UNCHANGED) for label in range(11, 0, -1)})
FRUITS_TEMPLATE_VAR_ANGLE_YOUTUBE = OrderedDict({label: [rotate_img(fruit, angle, 1.0) for angle in range(0, 360, 360//FRUITS_ROTATION[label])] for label, fruit in FRUITS_TEMPLATE_YOUTUBE.items()})
FRUITS_TEMPLATE_NEXT_YOUTUBE = OrderedDict({label: [rotate_img(fruit, -45, 1.0)] for label, fruit in FRUITS_TEMPLATE_YOUTUBE.items() if label <= 5})


def get_state(field, haved_area, next_area, fruits_template, fruits_template_var_angle, fruits_template_next, lower_obs, higher_obs):
	# field
	field_fruits_obs = list()
	for label in range(11, 0, -1):
		field, fruit_locations = detect_xy(
			fruits_template_var_angle[label],
			field,
			threshold=FRUITS_THRESHOLD[label],
			delete_match=True,
		)

		if fruit_locations.size == 0:
			continue

		fruit_state = np.hstack((fruit_locations, np.full((fruit_locations.shape[0], 1), label)))
		field_fruits_obs.extend(fruit_state)

	field_fruits_obs = sorted(field_fruits_obs, key=lambda x: x[0])[::-1]
	# limit MAX_FRUIT_NUM
	field_fruits_obs = field_fruits_obs[:MAX_FRUIT_NUM]
	field_fruits_obs = np.array(field_fruits_obs)
	field_state = calc_obs_space(field_fruits_obs, lower_obs, higher_obs)
	fruits_count = field_state.shape[0]

	# haved_fruit
	haved_fruit_label = 1.0
	for label in range(5, 0, -1):
		_, fruit_locations = detect_xy(
			[fruits_template[label]],
			haved_area,
			threshold=FRUITS_THRESHOLD_HIGH[label]
		)
		if fruit_locations.size:
			haved_fruit_label = float(label)
			break

	# next_fruit
	next_fruit_label = 1.0
	for label in range(5, 0, -1):
		_, fruit_locations = detect_xy(
			fruits_template_next[label],
			next_area,
			threshold=FRUITS_THRESHOLD_HIGH[label],
		)
		if fruit_locations.size:
			next_fruit_label = float(label)
			break

	meta_data = np.array([haved_fruit_label, next_fruit_label, fruits_count])

	state_data = np.zeros((61, 3))
	state_data[:fruits_count, :] = field_state
	state_data[-1, :] = meta_data
	return state_data


def get_state_from_ipad(img):
	# field = sc[341:986, 107:625, :] # obs_area
	field = img[409:986, 107:625, :]  # limit matching locations
	haved_area = img[260:370, 50:680, :]
	next_area = img[68:138, 408:450, :]
	return get_state(field, haved_area, next_area, FRUITS_TEMPLATE_IPAD, FRUITS_TEMPLATE_VAR_ANGLE_IPAD, FRUITS_TEMPLATE_NEXT_IPAD, (341 - 409, 0), field.shape[:2])


def get_state_from_youtube(img):
	# field = sc[115:673, 418:864, :] # obs_area
	field = img[174:673, 418:864, :]  # limit matching locations
	haved_area = img[40:145, 360:900, :]
	next_area = img[126:236, 1036:1130, :]
	return get_state(field, haved_area, next_area, FRUITS_TEMPLATE_YOUTUBE, FRUITS_TEMPLATE_VAR_ANGLE_YOUTUBE, FRUITS_TEMPLATE_NEXT_YOUTUBE, (115 - 174, 0), field.shape[:2])


if __name__ == '__main__':
	# active_window()
	# sc = get_screen()
	# sc = cv2.cvtColor(np.array(sc, dtype=np.uint8), cv2.COLOR_RGBA2BGR)
	# state = get_state_from_ipad(sc)
	image_files = get_image_files_path("../materials/images")
	image = cv2.imread(image_files[0])
	state = get_state_from_youtube(image)
	np.set_printoptions(precision=3, suppress=True)
	print(state)
