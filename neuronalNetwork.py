import activationFunctions as act
import random
import numpy as np
from collections.abc import Iterable

class NeuronalNetwork:
    weightMin = -1
    weightMax = 1
    biasMin = -10
    biasMax = 10
    
    def __init__(self, nbInput, nbOutput, outputFunction=act.sigmoidRelu) -> None:
        weight, bias = self.makeLayer(nbInput, nbOutput)
        self.layers = [nbInput, nbOutput]
        self.weights = [weight]
        self.biases = [bias]
        self.functions = [outputFunction]
        self.nbLayer = 2
    
    def addLayer(self, nbNode, function=act.relu):
        weight1, bias1 = self.makeLayer(self.layers[-2], nbNode)
        weight2, bias2 = self.makeLayer(nbNode, self.layers[-1])
        self.weights[-1] = weight1
        self.weights.append(weight2)
        self.biases[-1] = bias1
        self.biases.append(bias2)
        self.layers.insert(-1, nbNode)
        self.functions.insert(-1, function)
        self.nbLayer += 1
    
    def makeLayer(self, nbInput, nbOutput):
        weight = np.array([[random.uniform(self.weightMin, self.weightMax) for _ in range(nbInput)] for _ in range(nbOutput)])
        bias = np.array([random.uniform(self.biasMin, self.biasMax) for _ in range(nbOutput)]).reshape(nbOutput, 1)
        return weight, bias
    
    def calcul(self, *inputs):
        if isinstance(inputs[0], Iterable):
            inputs = np.array(inputs[0]).reshape(self.layers[0], 1)
        else:
            inputs = np.array(inputs).reshape(self.layers[0], 1)
        for i in range(self.nbLayer - 1):
            layerSize = self.layers[i+1]
            weight = self.weights[i]
            # print("weight :", weight)
            bias = self.biases[i]
            inputs = weight.dot(inputs)
            inputs = inputs + bias
            # print("inputs 1 :", inputs)
            inputs = np.array(list(map(self.functions[i], inputs.reshape((layerSize,))))).reshape((layerSize, 1))
            # print("inputs 2 :", inputs)
        return inputs
    
    def save(self, file):
        raise NotImplementedError()
    
    @staticmethod
    def load(file):
        raise NotImplementedError()

