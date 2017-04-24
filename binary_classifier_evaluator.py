#! /usr/bin/python3
import itertools
import os
import time

import numpy as np
import png

from area import Histogram
from data_types import *
from image import Image
import binary_classifier_builder as bcb

class Evaluator():

	context = {
		"__builtins__" : None,
		"add": Floats.add, "sub": Floats.sub, "mul": Floats.mul, "div": Floats.div,
		"add2": Floats2.add2, "sub2": Floats2.sub2, "mul2": Floats2.mul2, "div2": Floats2.div2,
		"add3": Floats3.add3, "sub3": Floats3.sub3, "mul3": Floats3.mul3, "div3": Floats3.div3,
		"bins": bins, "bins1": bins1, "bins2": bins2, "bins3": bins3,
		"distance": distance, "distance1": distance1, "distance2": distance2, "distance3": distance3,
		"HoG": HoG,
		"Image": Image, "Shape": Shape, "Position": Position, "Size": Size, "Histogram": Histogram, "Index": Index
	}

	def __init__(self, code, debug=False):
		evauluated_expression = 'lambda IN0: ' + code
		self.classificator = eval(evauluated_expression, self.context)
		self.debug = debug

	def lower(self, a):
		return a < 0

	def greater(self, a):
		return a > 0

	def count_correctly_classified(self, images_dir, comparison_operator):
		image_counter = 0
		correctly_classified_image_counter = 0

		if self.debug:
			print('Load {} test images for class'.format(len(os.listdir(images_dir))))

		for filename in os.listdir(images_dir):
			image = Image(os.path.join(images_dir, filename))

			if self.debug:
				print(str(filename) + ',' + str(self.classificator(image)))

			image_counter += 1
			if comparison_operator(self.classificator(image)):
				correctly_classified_image_counter += 1

		return (correctly_classified_image_counter, image_counter)

	def classify_pair_of_class(self, dir_with_images_from_negative_class, dir_with_images_from_positive_class):
		(correctly_classified_negative_image_counter, negative_image_counter) = self.count_correctly_classified(dir_with_images_from_negative_class, self.lower)
		(correctly_classified_positive_image_counter, positive_image_counter) = self.count_correctly_classified(dir_with_images_from_positive_class, self.greater)

		return (correctly_classified_negative_image_counter / negative_image_counter,
				correctly_classified_positive_image_counter / positive_image_counter,
				(correctly_classified_negative_image_counter + correctly_classified_positive_image_counter) / (positive_image_counter + negative_image_counter))


training_images_dir = '../../Dane/image_classification_datasets/motion_tracking/training/'
testing_images_dir = '../../Dane/image_classification_datasets/motion_tracking/testing/'
max_classs_id = 40

def evaluate_classificator(negative_class_subdir, positive_class_subdir):
	negative_class_train_dir_path = os.path.join(training_images_dir, str(negative_class_subdir))
	positive_class_train_dir_path = os.path.join(training_images_dir, str(positive_class_subdir))
	negative_class_test_dir_path = os.path.join(testing_images_dir, str(negative_class_subdir))
	positive_class_test_dir_path = os.path.join(testing_images_dir, str(positive_class_subdir))

	t = time.time()
	pop, stats, hof = bcb.generate_classificator(negative_class_train_dir_path, positive_class_train_dir_path)
	elapsed = time.time() - t

	evaluator = Evaluator(str(hof[0]))

	(neagtive_class_correctly_classified_stat, positive_class_correctly_classified_stat, both_class_correctly_classified_stat) = evaluator.classify_pair_of_class(negative_class_test_dir_path, positive_class_test_dir_path)

	print(str(training_images_dir) + ';' +  str(negative_class_subdir) + ';' + \
		  str(positive_class_subdir) +';' + str(elapsed) + ';' + \
		  str(stats.compile(pop)) + ';' + str(hof[0]) + ';' + \
		  str(neagtive_class_correctly_classified_stat) + ';' + \
		  str(positive_class_correctly_classified_stat) + ';' + \
		  str(both_class_correctly_classified_stat))


if __name__ == "__main__":
	bcb.prepare_genetic_tree_structure()

	for combination in list(itertools.combinations(range(max_classs_id), 2)):
		negative_class_subdir = combination[0]
		positive_class_subdir = combination[1]
		evaluate_classificator(negative_class_subdir, positive_class_subdir)
