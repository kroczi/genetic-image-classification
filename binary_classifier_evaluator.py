#! /usr/bin/python3
try:
	import configparser
except ImportError:
	import ConfigParser as configparser
import itertools
import logging
import os
import time

import numpy as np
import png

from area import Histogram
from data_types import *
from image import Image
import binary_classifier_builder as bcb

class ConfigSection:
    def __init__(self, config, section):
        self.config = config
        self.section = section

    def get(self, key):
        return self.config.get(self.section, key)

    def getint(self, key):
        return self.config.getint(self.section, key)

    def getfloat(self, key):
        return self.config.getfloat(self.section, key)

    def getboolean(self, key):
        return self.config.getboolean(self.section, key)


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

		logger.debug('Loading {} test images for class'.format(len(os.listdir(images_dir))))

		for filename in os.listdir(images_dir):
			image = Image(os.path.join(images_dir, filename))

			logger.debug(str(filename) + ',' + str(self.classificator(image)))

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



def evaluate_classificator(dataset_config, parameters_config, negative_class_subdir, positive_class_subdir):
	negative_class_train_dir_path = os.path.join(dataset_config.get("base_directory"), dataset_config.get("training_directory"), str(negative_class_subdir))
	positive_class_train_dir_path = os.path.join(dataset_config.get("base_directory"), dataset_config.get("training_directory"), str(positive_class_subdir))
	negative_class_test_dir_path = os.path.join(dataset_config.get("base_directory"), dataset_config.get("testing_directory"), str(negative_class_subdir))
	positive_class_test_dir_path = os.path.join(dataset_config.get("base_directory"), dataset_config.get("testing_directory"), str(positive_class_subdir))

	t = time.time()
	pop, stats, hof = bcb.generate_classificator(parameters_config, negative_class_train_dir_path, positive_class_train_dir_path)
	elapsed = time.time() - t

	evaluator = Evaluator(str(hof[0]))

	(negative_class_correctly_classified_stat, positive_class_correctly_classified_stat, both_class_correctly_classified_stat) = evaluator.classify_pair_of_class(negative_class_test_dir_path, positive_class_test_dir_path)

	logger.info(str(dataset_config.get("base_directory")) + ';' +  str(negative_class_subdir) + ';' + \
		  		str(positive_class_subdir) +';' + str(elapsed) + ';' + \
		  		str(stats.compile(pop)) + ';' + str(hof[0]) + ';' + \
		  		str(negative_class_correctly_classified_stat) + ';' + \
		  		str(positive_class_correctly_classified_stat) + ';' + \
		  		str(both_class_correctly_classified_stat))


def setup_logging():
	global logger
	logger = logging.getLogger('ic')
	logger.setLevel(logging.DEBUG)

	console_handler = logging.StreamHandler()
	console_handler.setLevel(logging.WARNING)
	console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
	logger.addHandler(console_handler)

	info_file_handler = logging.FileHandler("info.log")
	info_file_handler.setLevel(logging.INFO)
	info_file_handler.setFormatter(logging.Formatter('%(message)s'))
	logger.addHandler(info_file_handler)

	debug_file_handler = logging.FileHandler("debug.log")
	debug_file_handler.setLevel(logging.DEBUG)
	debug_file_handler.setFormatter(logging.Formatter('%(message)s'))
	logger.addHandler(debug_file_handler)


def acquire_configuration(dataset_config_file, parameters_config_file, dataset_profile, parameters_profile=None):
	dataset_configuration = configparser.ConfigParser()
	dataset_configuration.read(dataset_config_file)
	if dataset_configuration.has_section(dataset_profile):
		logger.debug("Loading dataset configuration for profile: " + dataset_profile)
		dataset_config = ConfigSection(dataset_configuration, dataset_profile)
	else:
		logger.error("!!!Dataset configuration not found!!!")
		raise KeyError

	parameters_configuration = configparser.ConfigParser()
	parameters_configuration.read(parameters_config_file)
	if parameters_configuration.has_section(parameters_profile):
		logger.debug("Loading parameters configuration for profile: " + parameters_profile)
		parameters_config = ConfigSection(parameters_configuration, parameters_profile)
	else:
		logger.debug("Loading parameters configuration from defaults.")
		parameters_config = ConfigSection(parameters_configuration, 'DEFAULT')

	return (dataset_config, parameters_config)


if __name__ == "__main__":
	DATASET_CONFIG_FILE = 'dataset_config.ini'
	PARAMETERS_CONFIG_FILE = 'parameters_config.ini'
	DATASET_PROFILE = ["MOTION_TRACKING", "MNIST"]
	PARAMETERS_PROFILE = ["MOTION_TRACKING_PARAMETERS", "MNIST_PARAMETERS"]

	setup_logging()

	for (dataset_profile, parameters_profile) in zip(DATASET_PROFILE, PARAMETERS_PROFILE):
		(dataset_config, parameters_config) = acquire_configuration(DATASET_CONFIG_FILE, PARAMETERS_CONFIG_FILE, dataset_profile, parameters_profile)

		logger.info("Starting computations with following parameters configuration: " + str(PARAMETERS_PROFILE))
		logger.info(" and following dataset configuration: " + str(DATASET_PROFILE))

		bcb.prepare_genetic_tree_structure(dataset_config.getint("min_width"), dataset_config.getint("min_height"))

		for combination in list(itertools.combinations(range(dataset_config.getint("classes")), 2)):
			negative_class_subdir = combination[0]
			positive_class_subdir = combination[1]
			evaluate_classificator(dataset_config, parameters_config, negative_class_subdir, positive_class_subdir)
