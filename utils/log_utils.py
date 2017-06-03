#! /usr/bin/python3
import logging


def setup_logging():
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
