#! /usr/bin/python3
import itertools
import multiprocessing
import operator
import os
import random

from termcolor import colored
from deap import algorithms, base, creator, gp, tools
import numpy as np

from area import Histogram
from image import Image
from data_types import Floats, Floats2, Floats3, Shape, Position, Size, Index, \
                       HoG, bins1, bins2, bins3, distance1, distance2, distance3

DEBUG = True
MULTITHREAD = True
POOL_SIZE = 4
MIN_WIDTH = 28
MIN_HEIGHT = 28


def learn_rate(pop):
    return np.max(pop) / len(train_set)


def map_eval_result_to_string(result):
    if result:
         return colored('Passed:', 'green')
    return colored('Failed:', 'red')


def eval_classification(individual):
    if DEBUG:
        print(individual)

    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)

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


def prepare_genetic_tree_structure():
    global toolbox
    toolbox = base.Toolbox()

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
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)


def generate_classificator(negative_class_train_dir_path, positive_class_train_dir_path):
    #random.seed(11)
    global train_set
    train_set = []

    for filename in os.listdir(negative_class_train_dir_path):
        train_set.append(Image(os.path.join(negative_class_train_dir_path, filename), 0))
    for filename in os.listdir(positive_class_train_dir_path):
        train_set.append(Image(os.path.join(positive_class_train_dir_path, filename), 1))

    if DEBUG:
        print('Loaded {} images from {} class training dataset'.format(negative_class_train_dir_path, len(os.listdir(negative_class_train_dir_path))))
        print('Loaded {} images from {} class training dataset'.format(positive_class_train_dir_path, len(os.listdir(positive_class_train_dir_path))))

    toolbox.register("evaluate", eval_classification)

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
    stats.register("learn_rate", learn_rate)

    algorithms.eaSimple(pop, toolbox, 0.7, 0.4, 20, stats, halloffame=hof, verbose=False)

    toolbox.unregister("evaluate")
    toolbox.unregister("map")

    if MULTITHREAD:
       pool.close()

    return pop, stats, hof
