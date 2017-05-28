#! /usr/bin/python3
import itertools
import logging
import os
import time

from area import Histogram
import binary_classifier_builder as bcb
from config_utils import acquire_configuration
from data_types import *
from image import Image
from log_utils import setup_logging


logger = logging.getLogger('ic')


class Evaluator:
	context = {
		"__builtins__": None,
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

		logger.debug('Loading {} test images for class'.format(len(os.listdir(images_dir))))

		for filename in os.listdir(images_dir):
			image = Image(os.path.join(images_dir, filename))

			logger.debug(str(filename) + ',' + str(self.classificator(image)))

			image_counter += 1
			if comparison_operator(self.classificator(image)):
				correctly_classified_image_counter += 1

		return correctly_classified_image_counter, image_counter

	def classify_pair_of_class(self, dir_with_images_from_negative_class, dir_with_images_from_positive_class):
		(correctly_classified_negative_image_counter, negative_image_counter) = self.count_correctly_classified(
			dir_with_images_from_negative_class, self.lower)
		(correctly_classified_positive_image_counter, positive_image_counter) = self.count_correctly_classified(
			dir_with_images_from_positive_class, self.greater)

		return (correctly_classified_negative_image_counter / negative_image_counter,
				correctly_classified_positive_image_counter / positive_image_counter,
				(correctly_classified_negative_image_counter + correctly_classified_positive_image_counter) / (
					positive_image_counter + negative_image_counter))



def evaluate_classifier(dataset_config, parameters_config, negative_class_subdir, positive_class_subdir):
	negative_class_train_dir_path = os.path.join(dataset_config.get("base_directory"), dataset_config.get("training_directory"), str(negative_class_subdir))
	positive_class_train_dir_path = os.path.join(dataset_config.get("base_directory"), dataset_config.get("training_directory"), str(positive_class_subdir))
	negative_class_test_dir_path = os.path.join(dataset_config.get("base_directory"), dataset_config.get("testing_directory"), str(negative_class_subdir))
	positive_class_test_dir_path = os.path.join(dataset_config.get("base_directory"), dataset_config.get("testing_directory"), str(positive_class_subdir))

	t = time.time()
	pop, stats, hof = bcb.generate_classificator(parameters_config, negative_class_train_dir_path,
												 positive_class_train_dir_path)
	elapsed = time.time() - t

	evaluator = Evaluator(str(hof[0]))

	(negative_class_correctly_classified_stat, positive_class_correctly_classified_stat, both_class_correctly_classified_stat)\
		= evaluator.classify_pair_of_class(negative_class_test_dir_path, positive_class_test_dir_path)

	logger.info(str(dataset_config.get("base_directory")) + ';' + str(negative_class_subdir) + ';' +
				str(positive_class_subdir) + ';' + str(elapsed) + ';' +
				str(stats.compile(pop)) + ';' + str(hof[0]) + ';' +
				str(negative_class_correctly_classified_stat) + ';' +
				str(positive_class_correctly_classified_stat) + ';' +
				str(both_class_correctly_classified_stat))


if __name__ == "__main__":
	DATASET_CONFIG_FILE = 'dataset_config.ini'
	PARAMETERS_CONFIG_FILE = 'parameters_config.ini'
	DATASET_PROFILE = ["MOTION_TRACKING", "MNIST"]
	PARAMETERS_PROFILE = ["MOTION_TRACKING_PARAMETERS", "MNIST_PARAMETERS"]

	setup_logging()
	positionGenerator = PositionGenerator()
	sizeGenerator = SizeGenerator()
	bcb.prepare_genetic_tree_structure(positionGenerator, sizeGenerator)

	for (dataset_profile, parameters_profile) in zip(DATASET_PROFILE, PARAMETERS_PROFILE):
		(dataset_config, parameters_config) = acquire_configuration(DATASET_CONFIG_FILE, PARAMETERS_CONFIG_FILE, dataset_profile, parameters_profile)

		logger.info("Starting computations with following parameters configuration: " + str(PARAMETERS_PROFILE))
		logger.info(" and following dataset configuration: " + str(DATASET_PROFILE))

		positionGenerator.setNewBorders(dataset_config.getint("min_width"), dataset_config.getint("min_height"))
		sizeGenerator.setNewBorders(dataset_config.getint("min_width"), dataset_config.getint("min_height"))

		for combination in list(itertools.combinations(range(dataset_config.getint("classes")), 2)):
			negative_class_subdir = combination[0]
			positive_class_subdir = combination[1]
			evaluate_classifier(dataset_config, parameters_config, negative_class_subdir, positive_class_subdir)
