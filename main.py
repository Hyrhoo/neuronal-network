from activationFunctions import *
import neuralNetwork as nNet
from random import uniform, random
import pickle

# import keras
# (train_X, train_y), (test_X, test_y) = keras.datasets.mnist.load_data()

# import tensorflow as tf
# print(tf.contrib.learn)
# from sklearn import datasets
# digits = datasets.load_digits()
# print(digits["DESCR"])


def showData(net: nNet.NeuralNetwork):
    nbweights = 0
    nbbiases = 0
    for weights in net.weights:
        shape = weights.shape
        nbweights += shape[0] * shape[1]

    for biases in net.biases:
        nbbiases += biases.shape[0]

    print(nbbiases, nbweights)


net = nNet.NeuralNetwork(728, 10)
net.addLayer(16, relu)
net.addLayer(16, relu)
# net = neuralNetwork.NeuralNetwork.loadB("saveData/net.pkl")
# net = neuralNetwork.NeuralNetwork.load("saveData/net.pkl")

showData(net)
datas = [random() for _ in range(net.inputSize)]
print(net.calcul(datas))

# net.save("dataSave/net.pkl")
# net.saveB("dataSave/net.pkl")











