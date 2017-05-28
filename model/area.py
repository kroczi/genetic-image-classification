#! /usr/bin/python3
import numpy as np


class Histogram:
	def __init__(self):
		self.array = np.zeros(8, np.float64)

	def add(self, index, magnitude):
		self.array[index] += magnitude

	def get(self, index):
		return self.array[index]

	def normalize(self):
		histogram_sum = self.array.sum()
		if histogram_sum != 0:
			self.array = np.array([x/histogram_sum for x in self.array])


class Area:
	def __init__(self, image):
		self.image = image

	def is_inside(self, x, y):
		raise NotImplementedError


class RectangleArea(Area):
	def __init__(self, image, left, top, width, height):
		Area.__init__(self, image)
		self.left = left % self.image.width
		self.top = top % self.image.height
		self.right = min(width + self.left, self.image.width)
		self.bottom = min(height + self.top, self.image.height)

	def is_inside(self, x, y):
		return (x >= self.left and x < self.right) and (y >= self.top and y < self.bottom)

	def create_histogram(self):
		histogram = Histogram()

		for y in range(self.top, self.bottom):
			for x in range(self.left, self.right):
				histogram.add(self.image.gradients[y][x].lower_bin, self.image.gradients[y][x].lower_weight)
				histogram.add(self.image.gradients[y][x].upper_bin, self.image.gradients[y][x].upper_weight)

		histogram.normalize()
		return histogram


class CircleArea(Area):
	def __init__(self, image, center_x, center_y, radius):
		Area.__init__(self, image)
		self.center_x = center_x % self.image.width
		self.center_y = center_y % self.image.height
		self.radius = radius

	def is_inside(self, x, y):
		return (x >= 0 and x < self.image.width) and (y >= 0 and y < self.image.height) and (x - self.center_x)**2 + (y - self.center_y)**2 <= self.radius**2

	def create_histogram(self):
		histogram = Histogram()

		for y in range(int(self.center_x - self.radius/2), int(self.center_x + self.radius/2)):
			for x in range(int(self.center_y - self.radius/2), int(self.center_y + self.radius/2)):
				if self.is_inside(x, y):
					histogram.add(self.image.gradients[y][x].lower_bin, self.image.gradients[y][x].lower_weight)
					histogram.add(self.image.gradients[y][x].upper_bin, self.image.gradients[y][x].upper_weight)

		histogram.normalize()
		return histogram
