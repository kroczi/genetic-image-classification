#! /usr/bin/python3
import math
import random

from area import RectangleArea, CircleArea


class Shape:
	def __init__(self, value):
		self.value = value

	def is_rectangle(self):
		return self.value == 0

	def __repr__(self):
		return "Shape(" + str(self.value) + ")"

class Size:
	def __init__(self, width, height):
		self.width = width
		self.height = height

	def __repr__(self):
		return "Size(" + str(self.width) + ", " + str(self.height) + ")"

class SizeGenerator:
	def setNewBorders(self, maxWidth, maxHeight):
		self.maxWidth = maxWidth
		self.maxHeight = maxHeight

	def generate(self):
		return Size(random.randint(3, self.maxWidth), random.randint(3, self.maxHeight))

class Position:
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def __repr__(self):
		return "Position(" + str(self.x) + ", " + str(self.y) + ")"

class PositionGenerator:
	def setNewBorders(self, maxWidth, maxHeight):
		self.maxWidth = maxWidth
		self.maxHeight = maxHeight

	def generate(self):
		return Position(random.randint(0, self.maxWidth), random.randint(0, self.maxHeight))

class Index:
	def __init__(self, value):
		self.value = int(value)

	def __repr__(self):
		return "Index(" + str(self.value) + ")"

class Floats:
	def __init__(self, value):
		self.value = float(value)

	def __repr__(self):
		return "Floats(" + str(self.value) + ")"

	@staticmethod
	def add(a, b):
		return float(a.value + b.value)

	@staticmethod
	def sub(a, b):
		return float(a.value - b.value)

	@staticmethod
	def mul(a, b):
		return float(a.value * b.value)

	@staticmethod
	def div(a, b):
		try:
			return float(a.value / b.value)
		except ZeroDivisionError:
			return float(1)


class Floats2(Floats):
	def __init__(self, value):
		Floats.__init__(self, value)

	def __repr__(self):
		return "Floats2(" + str(self.value) + ")"

	@staticmethod
	def add2(a, b):
		return Floats(a.value + b.value)

	@staticmethod
	def sub2(a, b):
		return Floats(a.value - b.value)

	@staticmethod
	def mul2(a, b):
		return Floats(a.value * b.value)

	@staticmethod
	def div2(a, b):
		try:
			return Floats(a.value / b.value)
		except ZeroDivisionError:
			return Floats(1)


class Floats3(Floats2):
	def __init__(self, value):
		Floats2.__init__(self, value)

	def __repr__(self):
		return "Floats3(" + str(self.value) + ")"

	@staticmethod
	def add3(a, b):
		return Floats2(a.value + b.value)

	@staticmethod
	def sub3(a, b):
		return Floats2(a.value - b.value)

	@staticmethod
	def mul3(a, b):
		return Floats2(a.value * b.value)

	@staticmethod
	def div3(a, b):
		try:
			return Floats2(a.value / b.value)
		except ZeroDivisionError:
			return Floats2(1)


def HoG(image, shape, position, size):
	if shape.is_rectangle():
		area = RectangleArea(image, position.x, position.y, size.width, size.height)
	else:
		area = CircleArea(image, position.x, position.y, min(size.width, size.height))

	return area.create_histogram()


def bins(histogram, index):
	return histogram.get(index.value)

def bins1(histogram, index):
	return Floats(bins(histogram, index))

def bins2(histogram, index):
	return Floats2(bins(histogram, index))

def bins3(histogram, index):
	return Floats3(bins(histogram, index))


def distance(histogram_a, histogram_b):
	sum = 0
	for (a, b) in zip(histogram_a.array, histogram_b.array):
		sum += (a - b)**2
	return math.sqrt(sum)

def distance1(histogram_a, histogram_b):
	return Floats(distance(histogram_a, histogram_b))

def distance2(histogram_a, histogram_b):
	return Floats2(distance(histogram_a, histogram_b))

def distance3(histogram_a, histogram_b):
	return Floats3(distance(histogram_a, histogram_b))
