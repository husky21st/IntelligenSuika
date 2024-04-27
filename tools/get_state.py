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

FRUITS_THRESHOLD_HIGH = dict((
	(1, 0.989),
	(2, 0.987),
	(3, 0.987),
	(4, 0.996),
	(5, 0.997),
))

FRUITS_TEMPLATE_IPAD = OrderedDict({label: cv2.imread(f"./../assets/f{str(label)}.png", cv2.IMREAD_UNCHANGED) for label in range(11, 0, -1)})


def get_state(field, haved_area, next_area, fruits_template, lower_obs, higher_obs):
	fruits_template_var_angle = OrderedDict({label: [rotate_img(fruit, angle, 1.0) for angle in range(0, 360, 360//FRUITS_ROTATION[label])] for label, fruit in fruits_template.items()})
	fruits_template_next = OrderedDict({label: [cv2.resize(rotate_img(fruit, -45, 1.0), dsize=None, fx=0.4, fy=0.4)] for label, fruit in fruits_template.items() if label <= 5})
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


def get_state_from_ipad():
	sc = get_screen()
	sc = cv2.cvtColor(np.array(sc, dtype=np.uint8), cv2.COLOR_RGBA2BGR)
	# field = sc[335:986, 107:625, :] # obs_area
	field = sc[409:986, 107:625, :]  # limit matching locations
	haved_area = sc[260:370, 50:680, :]
	next_area = sc[68:138, 408:450, :]
	return get_state(field, haved_area, next_area, FRUITS_TEMPLATE_IPAD, (-59, 0), field.shape[:2])


if __name__ == '__main__':
	active_window()
	state = get_state_from_ipad()
	np.set_printoptions(precision=3, suppress=True)
	print(state)
