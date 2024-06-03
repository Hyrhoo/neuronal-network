from node import Node
from activationFunctions import *

from random import uniform, random
import neuronalNetwork

import keras

# import tensorflow as tf
# print(tf.contrib.learn)
# from sklearn import datasets
# digits = datasets.load_digits()
# print(digits["DESCR"])


# myNode = Node(gelu, random())
# data = list(uniform(-100, 100) for _ in range(10))
# print(myNode.activate(*data))
# myNode.activationFunction = indently
# print(myNode.activate(*data))

# print(Node.loadString("indently,1.93279").activate(*(uniform(-100, 100) for _ in range(1000))))


net = neuronalNetwork.NeuronalNetwork(728, 10, sigmoid)
net.addLayer(16, relu)
net.addLayer(16, relu)

nbweights = 0
nbbiases = 0
for weights in net.weights:
    shape = weights.shape
    nbweights += shape[0] * shape[1]

for biases in net.biases:
    nbbiases += biases.shape[0]

print(nbbiases, nbweights)

print(net.calcul([random() for i in range(728)]))
