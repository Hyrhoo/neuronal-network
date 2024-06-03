from node import Node
from activationFunctions import relu, gelu, indently, softplus

from random import uniform, random
import math
import neuronalNetwork

# myNode = Node(gelu, random())
# data = list(uniform(-100, 100) for _ in range(10))
# print(myNode.activate(*data))
# myNode.activationFunction = indently
# print(myNode.activate(*data))

# print(Node.loadString("indently,1.93279").activate(*(uniform(-100, 100) for _ in range(1000))))

# print(math.sqrt(784))

net = neuronalNetwork.NeuronalNetwork(10, 2)
net.addLayer(50, gelu)
net.addLayer(100, relu)
net.addLayer(25, softplus)
print(net.calcul(1, 0, 0, 0, 0, 0, 0, 0, 0, 0))