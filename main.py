from node import Node
from activationFunctions import gelu, indently

from random import uniform, random
import math

# myNode = Node(gelu, random())
# data = list(uniform(-100, 100) for _ in range(10))
# print(myNode.activate(*data))
# myNode.activationFunction = indently
# print(myNode.activate(*data))

# print(Node.loadString("indently,1.93279").activate(*(uniform(-100, 100) for _ in range(1000))))

print(math.sqrt(784))