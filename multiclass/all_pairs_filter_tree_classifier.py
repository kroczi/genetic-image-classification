#! /usr/bin/python3
import os

from model.data_types import Floats, Floats2, Floats3, Shape, Position, Size, Index, \
	HoG, bins1, bins2, bins3, distance1, distance2, distance3
from model.image import Image
from model.area import Histogram

context = {
	"__builtins__": None,
	"add": Floats.add, "sub": Floats.sub, "mul": Floats.mul, "div": Floats.div,
	"add2": Floats2.add2, "sub2": Floats2.sub2, "mul2": Floats2.mul2, "div2": Floats2.div2,
	"add3": Floats3.add3, "sub3": Floats3.sub3, "mul3": Floats3.mul3, "div3": Floats3.div3,
	"bins1": bins1, "bins2": bins2, "bins3": bins3,
	"distance1": distance1, "distance2": distance2, "distance3": distance3,
	"HoG": HoG,
	"Image": Image, "Shape": Shape, "Position": Position, "Size": Size, "Histogram": Histogram, "Index": Index
}


def get_classifiers(classifiers_file):
	with open(classifiers_file) as classifiers:
		return [eval('lambda IN0: ' + x.strip(), context) for x in classifiers.readlines()]


def get_classifier_between(i, j, classes, classifiers):
	if i > j:
		(i, j) = (j, i)

	return classifiers[int((classes - 1 + classes - i) * i / 2 + (j - i - 1))]


def apft_classification(image, classes, classifiers):
	current_queue = []
	for i in range(classes):
		current_queue.append(i)

	while len(current_queue) > 1:
		next_queue = []
		while len(current_queue) > 0:
			i = current_queue.pop(0)
			if len(current_queue) > 0:
				j = current_queue.pop(0)
				if get_classifier_between(i, j, classes, classifiers)(image) > 0:
					next_queue.append(j)
				else:
					next_queue.append(i)
			else:
				next_queue.append(i)

		current_queue = next_queue

	return current_queue[0]


def apft_classification_of_whole_dataset(base_directory, classes, classifiers):
	for c in range(classes):
		for directory in os.listdir(base_directory):
			if os.path.isdir(os.path.join(base_directory, directory)):
				for filename in os.listdir(os.path.join(base_directory, directory, str(c))):
					path = os.path.join(base_directory, directory, str(c), filename)
					image = Image(path, c)
					result = apft_classification(classifiers, image, classifiers)
					print(str(path) + ', ' + str(c) + ', ' + str(result))


if __name__ == "__main__":
	CLASSES = 41
	CLASSIFIERS_FILE = "../../Dane/image_classification_datasets/motion_tracking/motion_tracking_classifiers.txt"
	BASE_DIRECTORY = '../../Dane/image_classification_datasets/motion_tracking/'

	classifiers = get_classifiers(CLASSIFIERS_FILE)
	apft_classification_of_whole_dataset(BASE_DIRECTORY, CLASSES, classifiers)

# IMAGE_FILE = "../../Dane/image_classification_datasets/motion_tracking/training/30/0021.png"
# image = Image(IMAGE_FILE)
# result = apft_classification(classifiers, image)
# print(str(IMAGE_FILE) + ', ' + str(result))
