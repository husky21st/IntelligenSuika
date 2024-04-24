import cv2
import os
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from typing import Union

button_template = cv2.imread("../assets/button.png")

# 80 slices color template
FRUITS_BGR_AND_NUM = OrderedDict((
	("3", ((245, 90, 165), 5)),
	("5", ((32, 131, 244), 10)),
	("4", ((14, 188, 246), 8)),
	("1", ((47, 35, 245), 5)),
	("2", ((82, 97, 242), 3)),
))

FRUITS_BAR_Y = dict((
	("1", 111),
	("2", 114),
	("3", 128),
	("4", 129),
	("5", 138),
))

FRUITS_DELAY_FRAME = dict((
	("1", 3),
	("2", 4),
	("3", 5),
	("4", 6),
	("5", 7),
))


def is_game_scene(frame):
	button_area = frame[700:, 440:490]
	mse, se = cv2.quality.QualityMSE_compute(button_area, button_template)
	if sum(mse) >= 1000:
		return False
	else:
		return True


def haved_fruit_check(frame) -> Union[None, str]:
	drop_fruit_area = frame[80, 452:820, :]
	for label, (bgr, num) in FRUITS_BGR_AND_NUM.items():
		mse = np.mean(np.square(drop_fruit_area - bgr), axis=1)
		if np.mean(np.partition(mse.ravel(), num)[:num]) < 100:
			return label
	return None


def exist_bar_check(frame, fruit_label) -> bool:
	gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	check_y = FRUITS_BAR_Y[fruit_label]
	bar_area = gray_frame[check_y:check_y + 7, 452:820]
	if max(bar_area.mean(axis=0)) < 240:
		# overlap check
		background_area = gray_frame[check_y + 7, 460:812]
		if np.all((206 <= background_area) & (background_area <= 224)):
			return False
		else:
			return True
	else:
		return True


def save_drop_fruit_scene(frame, prev_haved_fruit, prev_exist_bar, movie_name) -> (int, bool, bool):
	# game scene check
	if not is_game_scene(frame):
		return 30, None, True

	if prev_haved_fruit is None:
		# fruit check
		if prev_exist_bar:
			return 1, haved_fruit_check(frame), True
		else:
			fruit = haved_fruit_check(frame)
			if fruit is None:
				return 1, None, False
			else:
				return 1, fruit, True
	else:
		# bar check
		bar_flag = exist_bar_check(frame, prev_haved_fruit)
		if bar_flag:
			return 1, prev_haved_fruit, True
		else:
			# save scene
			cv2.imwrite(f"./images/{movie_name}/{str(i).zfill(8)}_{prev_haved_fruit}.png", frame)
			return FRUITS_DELAY_FRAME[prev_haved_fruit], None, False


if __name__ == '__main__':
	os.makedirs("./images", exist_ok=True)
	files = os.listdir("./movies")
	for file in tqdm(files):
		if file[-4:] != ".mp4":
			continue
		cap = cv2.VideoCapture(os.path.join("./movies", file))
		if not cap.isOpened():
			break
		movie_name = file[:-4]
		os.makedirs(f"./images/{movie_name}", exist_ok=True)

		fps = cap.get(cv2.CAP_PROP_FPS)
		num_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
		haved_fruit = None
		exist_bar = True
		i = 0
		pbar = tqdm(range(int(num_frame)), desc="movie frame process", ncols=80, total=num_frame)
		while i < num_frame:
			cap.set(cv2.CAP_PROP_POS_FRAMES, i)
			ret, frame = cap.read()
			if ret:
				frame_skip, haved_fruit, exist_bar = save_drop_fruit_scene(frame, haved_fruit, exist_bar,
																		   movie_name=movie_name)
				i += frame_skip
				pbar.update(frame_skip)
			else:
				i += 1
				pbar.update(frame_skip)
		pbar.close()
