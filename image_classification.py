#! /usr/bin/python3
import png
import itertools
import math
import multiprocessing
import numpy as np
import operator
import random

from termcolor import colored
from deap import algorithms
from deap import base
from deap import creator
from deap import gp
from deap import tools

DEBUG = False
MULTITHREAD = True
POOL_SIZE = 4
MIN_WIDTH = 128
MIN_HEIGHT = 128

toolbox = base.Toolbox()
image_set = []

class Gradient:
    def __init__(self, lower_bin, upper_bin, lower_weight, upper_weight):
        self.lower_bin = int(lower_bin)
        self.upper_bin = int(upper_bin)
        self.lower_weight = lower_weight
        self.upper_weight = upper_weight


class Image:
    def __init__(self, path, species=None):
        if DEBUG:
            print(path)
        reader = png.Reader(path)
        (w, h, p, m) = reader.read()
        self.width = w
        self.height = h
        self.array = np.array(list(p), np.int16)
        # The class of the image
        self.species = species
        self.gradients = [[self.calculate_gradient(x, y) for x in range(self.width)] for y in range(self.height)]

    def value(self, x, y):
        return self.array[y][x]

    def calculate_gradient(self, x, y):
        if x == 0 or x == self.width - 1 or y == 0 or y == self.height - 1:
            return Gradient(0.0, 0.0, 0.0, 0.0)

        gx = self.value(x+1, y) - self.value(x-1, y)
        gy = self.value(x, y+1) - self.value(x, y-1)
        magnitude = math.sqrt(gx**2 + gy**2)

        orientation = math.atan2(gy, gx) * 180.0 / math.pi
        if orientation < 0:
            orientation += 360.0

        lower_bin = int(orientation // 45.0) % 8
        upper_bin = int(lower_bin + 1)

        lower_weight = magnitude * (45.0 - (orientation - lower_bin * 45.0)) / 45.0
        upper_weight = magnitude * (45.0 - (upper_bin * 45.0 - orientation)) / 45.0

        return Gradient(lower_bin, upper_bin % 8, lower_weight, upper_weight)


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


class Position:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return "Position(" + str(self.x) + ", " + str(self.y) + ")"


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
        try: return float(a.value / b.value)
        except ZeroDivisionError: return float(1)


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
        try: return Floats(a.value / b.value)
        except ZeroDivisionError: return Floats(1)


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
        try: return Floats2(a.value / b.value)
        except ZeroDivisionError: return Floats2(1)


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

def map_eval_result_to_string(result):
    if result == True:
         return colored('Passed:', 'green')
    elif result == False:
         return colored('Failed:', 'red')

def eval_classification(individual):
    if DEBUG:
        print(individual)

    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)

    # Randomly sample 30 images
    #train_set = random.sample(image_set, 30)
    train_set = image_set

    # Evaluate the number of correctly classified images
    result = 0
    for image in train_set:
        outcome = func(image)
        correctly_classified = ((outcome > 0 and image.species == 1) or (outcome < 0 and image.species == 0))
        if (correctly_classified):
            result += 1
        if DEBUG:
            print(map_eval_result_to_string(correctly_classified)  + str(outcome))

    return result,


def plot_tree(individual):
    import pygraphviz as pgv

    nodes, edges, labels = gp.graph(individual)
    g = pgv.AGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    g.layout(prog="dot")

    for i in nodes:
        n = g.get_node(i)
        n.attr["label"] = labels[i]

    g.draw("tree.png")


def plot_tree2(individual):
    import matplotlib.pyplot as plt
    import networkx as nx

    nodes, edges, labels = gp.graph(individual)
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    pos = nx.random_layout(g)

    nx.draw_networkx_nodes(g, pos)
    nx.draw_networkx_edges(g, pos)
    nx.draw_networkx_labels(g, pos, labels)
    plt.show()


#################################################

def main():
    #random.seed(11)
    for i in range(0, 40):
        image_set.append(Image("../Dane/COIL20/obj1__" + str(i) + ".png", 0))
        image_set.append(Image("../Dane/COIL20/obj2__" + str(i) + ".png", 1))

    pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(Image, 1), float, "IN")

    pset.addPrimitive(Floats.add,   [Floats, Floats], float)
    pset.addPrimitive(Floats2.add2, [Floats2, Floats2], Floats)
    pset.addPrimitive(Floats3.add3, [Floats3, Floats3], Floats2)
    pset.addPrimitive(Floats.sub,   [Floats, Floats], float)
    pset.addPrimitive(Floats2.sub2, [Floats2, Floats2], Floats)
    pset.addPrimitive(Floats3.sub3, [Floats3, Floats3], Floats2)
    pset.addPrimitive(Floats.mul,   [Floats, Floats], float)
    pset.addPrimitive(Floats2.mul2, [Floats2, Floats2], Floats)
    pset.addPrimitive(Floats3.mul3, [Floats3, Floats3], Floats2)
    pset.addPrimitive(Floats.div,   [Floats, Floats], float)
    pset.addPrimitive(Floats2.div2, [Floats2, Floats2], Floats)
    pset.addPrimitive(Floats3.div3, [Floats3, Floats3], Floats2)
    #pset.addPrimitive(bins,  [Histogram, Index], float)
    pset.addPrimitive(bins1, [Histogram, Index], Floats)
    pset.addPrimitive(bins2, [Histogram, Index], Floats2)
    pset.addPrimitive(bins3, [Histogram, Index], Floats3)
    #pset.addPrimitive(distance,  [Histogram, Histogram], float)
    pset.addPrimitive(distance1, [Histogram, Histogram], Floats)
    pset.addPrimitive(distance2, [Histogram, Histogram], Floats2)
    pset.addPrimitive(distance3, [Histogram, Histogram], Floats3)
    pset.addPrimitive(HoG, [Image, Shape, Position, Size], Histogram)

    pset.addEphemeralConstant("shape", lambda: Shape(random.randint(0, 1)), Shape)
    pset.addEphemeralConstant("coords", lambda: Position(random.randint(0, MIN_WIDTH), random.randint(0, MIN_HEIGHT)), Position)
    pset.addEphemeralConstant("size", lambda: Size(random.randint(3, MIN_WIDTH), random.randint(3, MIN_HEIGHT)), Size)
    pset.addEphemeralConstant("index", lambda: Index(random.randint(0, 7)), Index)

    if DEBUG:
        print (pset.primitives)
        print (pset.terminals)

    pset.context['Position'] = Position
    pset.context['Shape'] = Shape
    pset.context['Size'] = Size
    pset.context['Index'] = Index

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=4)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("evaluate", eval_classification)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    if MULTITHREAD:
        pool = multiprocessing.Pool(POOL_SIZE)
        toolbox.register("map", pool.map)

    pop = toolbox.population(n=100)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    algorithms.eaSimple(pop, toolbox, 0.5, 0.2, 10, stats, halloffame=hof)

    expr = toolbox.individual()
    nodes, edges, labels = gp.graph(expr)

    return pop, stats, hof


if __name__ == "__main__":
    pop, stats, hof = main()
    print(stats.compile(pop))
    print(hof[0])
    plot_tree(hof[0])
