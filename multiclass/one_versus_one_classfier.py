#! /usr/bin/python3
import os

import numpy as np
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


class ClassifierWithPriority:
	def __init__(self, expression, priority=1):
		self.evaluator = eval('lambda IN0: ' + expression.strip(), context)
		self.priority = float(priority)


def get_classifiers(classifiers_file):
	with open(classifiers_file) as classifiers:
		return [ClassifierWithPriority(*(line.strip().split(";"))) for line in classifiers.readlines()]


def ovo_classification(image, classes, classifiers, indices):
	ranking = np.zeros((classes,))
	index = 0

	for i in range(classes):
		for j in range(i + 1, classes):
			if i in indices and j in indices:
				if classifiers[index].evaluator(image) > 0:
					ranking[j] += classifiers[index].priority
				else:
					ranking[i] += classifiers[index].priority
			index += 1

	return ranking


def ovo_classification_of_whole_dataset(base_directory, classes, classifiers, indices):
	for c in range(classes):
		for directory in os.listdir(base_directory):
			if os.path.isdir(os.path.join(base_directory, directory)):
				for filename in os.listdir(os.path.join(base_directory, directory, str(c))):
					path = os.path.join(base_directory, directory, str(c), filename)
					image = Image(path, c)
					ranking = ovo_classification(image, classes, classifiers, indices)
					print(str(path) + ', ' + str(c) + ', ' + str(ranking.argmax()))


if __name__ == "__main__":
	CLASSES = 41
	CLASSIFIERS_FILE = "../../Dane/image_classification_datasets/motion_tracking/motion_tracking_classifiers_improved_with_priorities.txt"
	BASE_DIRECTORY = '../../Dane/image_classification_datasets/motion_tracking/'

	classifiers = get_classifiers(CLASSIFIERS_FILE)
	indices = list(range(CLASSES))
	ovo_classification_of_whole_dataset(BASE_DIRECTORY, CLASSES, classifiers, indices)

# IMAGE_FILE = "../../Dane/image_classification_datasets/motion_tracking/training/30/0021.png"
# image = Image(IMAGE_FILE)
# indices = list(range(CLASSES))
# ranking = ovo_classification(classifiers, indices, image)
# print(str(IMAGE_FILE) + ', ' + str(ranking.argmax()))
