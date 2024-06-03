import neuralNetwork as nNet
import numpy as np
import random

class NetworkTrainer:

    MIX_POURCENTAGE = 0.7
    MIX_CHANCE = 0.5
    RANDOM_CHANGE = 0.1
    
    def __init__(self, nbNetwork, nbGeneration, layers, functions, weightRange=(-1.0, 1.0), biasRange=(-5.0, 5.0)) -> None:
        self.nbNetwork = nbNetwork
        self.nbGeneration = nbGeneration
        self.networks = []
        self.res = {}
        for _ in range(nbNetwork):
            self.networks.append(nNet.NeuralNetwork(layers, functions, weightRange, biasRange))
        
    def randomMutate(self, network:nNet.NeuralNetwork):
        net = network.copy
        for weight in net.weights:
            for x in np.nditer(weight, op_flags=['readwrite']):
                if random.random() < self.RANDOM_CHANGE:
                    x[...] = random.uniform(*net.weightRange)
        return net
    
    def mixMutate(self, network1, network2):
        raise NotImplementedError("mixMutate")
    
    def updateNetworks(self):
        raise NotImplementedError("updateNetworks")

    def train(self, datas, sempleSize):
        # main training methode
        for i in range(self.nbGeneration):
            semple = datas[i*sempleSize:(i+1)*sempleSize]
            for data in semple:
                self.calculeAll(data)
            self.computeScore()

    def calculeAll(self, data):
        raise NotImplementedError("calculeAll")
    
    def computeScore(self):
        raise NotImplementedError("computeScore")

