import numpy as np
import cv2
import os
from collections import OrderedDict


def make_dir(dir_path):
	if not os.path.exists(dir_path):
		os.mkdir(dir_path)


def rotate_img(image, angle, scale, border_value=(255, 255, 255)):
	height, width = image.shape[:2]
	center = (int(width / 2), int(height / 2))
	trans = cv2.getRotationMatrix2D(center, angle, scale)
	rot_image = cv2.warpAffine(image, trans, (width, height), borderValue=border_value)
	return rot_image


def plot_box(img, boxes):
	result_img = img.copy()
	for box in boxes:
		result_img = cv2.rectangle(
			result_img,
			pt1=box[:2],
			pt2=box[2:],
			color=(255, 0, 0),
			thickness=2)
	return result_img


def nms(boxes, scores, nms_thresh=0.5, top_k=200):
	"""
	boxes: np.array([[x1, y1, x2, y2],...])
	"""
	keep = []
	if len(boxes) == 0:
		return keep
	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	x2 = boxes[:, 2]
	y2 = boxes[:, 3]
	area = (x2 - x1) * (y2 - y1)
	idx = np.argsort(scores, axis=0)
	idx = idx[-top_k:]

	while len(idx) > 0:
		last = len(idx) - 1
		i = idx[last]  # index of current largest val
		keep.append(i)
		xx1 = np.maximum(x1[i], x1[idx[:last]])
		yy1 = np.maximum(y1[i], y1[idx[:last]])
		xx2 = np.minimum(x2[i], x2[idx[:last]])
		yy2 = np.minimum(y2[i], y2[idx[:last]])

		w = np.maximum(0, xx2 - xx1)
		h = np.maximum(0, yy2 - yy1)

		inter = w * h
		iou = inter / (area[idx[:last]] + area[i] - inter)
		idx = np.delete(idx, np.concatenate(
			([last], np.where(iou > nms_thresh)[0])))

	return boxes[keep], scores[keep]


def template_match(
		image,
		templs,
		method=cv2.TM_CCOEFF_NORMED,
		threshold=0.5,
):
	templ_l = templs[0].shape[0]
	templ_remove_l = int(templ_l * 0.46)
	res = np.empty((len(templs), image.shape[0] - templ_l + 1, image.shape[1] - templ_l + 1))
	for i, templ in enumerate(templs):
		res[i] = cv2.matchTemplate(image=image, templ=templ[:, :, :3], method=method, mask=templ[:, :, 3] // 255)

	res = np.nan_to_num(res.max(axis=0))
	#  delete error
	np.place(res, res > 1, 0)
	np.place(res, res < threshold, 0)

	all_found_idx = list()
	while res.max() != 0:
		max_idx = np.unravel_index(np.argmax(res), res.shape)
		all_found_idx.append(max_idx)
		res[max(max_idx[0] - templ_remove_l, 0):min(max_idx[0] + templ_remove_l, res.shape[0]),
		max(max_idx[1] - templ_remove_l, 0):min(max_idx[1] + templ_remove_l, res.shape[1])] = 0
	return np.array(all_found_idx)


def detect_xy(templates, img, threshold=0.2, nms_thresh=0.2):
	all_found_idx = template_match(
		image=img,
		templs=templates,
		method=cv2.TM_CCORR_NORMED,
		threshold=threshold
	)
	if all_found_idx.size == 0:
		return img, []

	fruit_locations = list()
	templ_l = templates[0].shape[0]
	delete_area = 1 - rotate_img(templates[0][:, :, 3], 0, 0.9, border_value=0) / 255
	delete_area = np.repeat(delete_area[:, :, None], 3, axis=2).astype(np.uint8)
	for found_idx in all_found_idx:
		# detect xy
		fruit_locations.append(found_idx + templ_l // 2)
		# delete fruit from img
		img[found_idx[0]:found_idx[0] + templ_l, found_idx[1]:found_idx[1] + templ_l, :] *= delete_area

	return img, fruit_locations
