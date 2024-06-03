from collections.abc import Iterable, Callable
import numpy as np
import pickle
import activationFunctions as actFn
from overload import overload
from copy import deepcopy

class NeuralNetwork:
    
    @overload
    def __init__(self, nbInput:int, nbOutput:int, weightRange:tuple=(-1.0, 1.0), biasRange:tuple=(-5.0, 5.0), outputFunction:Callable=actFn.sigmoidRelu) -> None:
        """
        The __init__ function initializes the neural network with a number of inputs, outputs, and an output function.
        It also sets the weight range and bias range to default values if none are given.
        
        
        :param self: Represent the instance of the class
        :param nbInput: Set the number of input neurons in the first layer
        :param nbOutput: Determine the number of neurons in the output layer
        :param weightRange: Set the range of values that the weights can take
        :param 1): Define the number of input neurons
        :param biasRange: Determine the range of values that the bias can take
        :param 10): Set the number of neurons in the output layer
        :param outputFunction: Specify the activation function of the output layer
        :return: None
        :doc-author: Trelent
        """
        self.weightRange = weightRange
        self.biasRange = biasRange
        self.functions = [outputFunction]
        self.layers = [nbInput, nbOutput]
        weight, bias = self.makeLayer(nbInput, nbOutput)
        self.weights = [weight]
        self.biases = [bias]
    
    @__init__.add
    def __init__(self, layers:Iterable, functions:Iterable, weightRange:tuple, biasRange:tuple) -> None:
        self.weights = []
        self.biases = []
        self.layers = layers
        self.functions = functions
        self.weightRange = weightRange
        self.biasRange = biasRange
        for i in range(len(layers) - 1):
            weight, bias = self.makeLayer(layers[i], layers[i+1])
            self.weights.append(weight)
            self.biases.append(bias)
    
    # @__init__.add
    # def __init__(self, weights:Iterable, biases:Iterable, functions:Iterable, weightRange:tuple, biasRange:tuple) -> None:
    #     self.weightRange = weightRange
    #     self.biasRange = biasRange
    #     self.weights = [np.array(i) for i in weights]
    #     self.biases = [np.array(i) for i in biases]
    #     self.functions = functions
    #     self.layers = [weight.shape[0] for weight in weights]
    #     self.layers.append(weights[-1].shape[1])
    
    @property
    def inputSize(self):
        return self.layers[0]
    
    @property
    def outputSize(self):
        return self.layers[-1]
    
    @property
    def nbLayer(self):
        return len(self.layers)
    
    @property
    def copy(self):
        return deepcopy(self)
    
    def addLayer(self, nbNode, function=actFn.relu) -> None:
        """
        The addLayer function adds a layer to the neural network.
            It takes in two arguments:
                nbNode (int): The number of nodes in the new layer.
                function (Callable): The activation function for the new layer. Defaults to actFn.relu().
        
        :param self: Access the attributes and methods of the class in python
        :param nbNode: Specify the number of nodes in the new layer
        :param function: Define the activation function of the layer
        :return: None
        :doc-author: Trelent
        """
        weight1, bias1 = self.makeLayer(self.layers[-2], nbNode)
        weight2, bias2 = self.makeLayer(nbNode, self.layers[-1])
        self.weights[-1] = weight1
        self.weights.append(weight2)
        self.biases[-1] = bias1
        self.biases.append(bias2)
        self.layers.insert(-1, nbNode)
        self.functions.insert(-1, function)
    
    def makeLayer(self, nbInput, nbOutput) -> None:
        """
        The makeLayer function creates a layer of neurons with random weights and biases.
            The number of inputs is the number of neurons in the previous layer, and the 
            number of outputs is equal to the desired size for this new layer. The weightRange 
            variable determines how large or small these weights can be, while biasRange does 
            so for biases.
        
        :param self: Represent the instance of the class
        :param nbInput: Determine the number of input neurons and the nboutput parameter is used to determine the number of output neurons
        :param nbOutput: Specify the number of neurons in the layer
        :return: A tuple of the weight and bias
        :doc-author: Trelent
        """
        weight = np.random.uniform(*self.weightRange, (nbInput, nbOutput))
        bias = np.random.uniform(*self.biasRange, (nbOutput,))
        return weight, bias
    
    def calcul(self, *inputs):
        """
        The calcul function takes in a list of inputs and returns the output of the neural network.
        The input is first multiplied by each weight matrix, then added to each bias vector, and finally passed through an activation function.
        This process is repeated for every layer in the neural network.
        
        :param self: Represent the instance of the class
        :param *inputs: Pass a list of inputs to the function
        :return: The output of the network for a given input
        :doc-author: Trelent
        """
        if isinstance(inputs[0], Iterable):
            inputs = np.array(inputs[0])
        else:
            inputs = np.array(inputs)

        for i in range(self.nbLayer - 1):
            # layerSize = self.layers[i+1]
            weight = self.weights[i]
            bias = self.biases[i]
            function = self.functions[i]
            inputs = inputs.dot(weight)
            inputs = inputs + bias
            inputs = np.array(list(map(function, inputs)))
        return inputs
    
    def toDict(self) -> dict:
        """
        The toDict function returns a dictionary containing all the information needed to recreate the neural network.
        The keys are:
            - ranges: A dictionary containing two keys, weight and bias, which contain tuples of length 2 representing the range in which weights and biases can be initialized.
            - functions: A list of strings representing activation functions for each layer (except input). The order is from first hidden layer to last output layer. 
                For example, if there are 3 layers with sigmoid as activation function for first two layers and softmax as activation function for last one then this key will have value [&quot;sigmoid&quot;, &quot;
        
        :param self: Refer to the object itself
        :return: A dictionary with the following keys:
        :doc-author: Trelent
        """
        datas = {}
        datas["ranges"] = {"weight": self.weightRange, "bias": self.biasRange}
        datas["functions"] = self.functions
        datas["layers"] = {"lst": self.layers, "nb": self.nbLayer}
        datas["weights"] = [weight.tolist() for weight in self.weights]
        datas["biases"] = [bias.tolist() for bias in self.biases]
        return datas

    def save(self, file) -> None:
        """
        The save function takes a file name as an argument and saves the current state of the object to that file.
            The save function uses pickle to serialize the object into a binary format, which is then written to disk.
        
        :param self: Represent the instance of the class
        :param file: Specify the name of the file to save to
        :return: The self
        :doc-author: Trelent
        """
        with open(file, "wb") as f:
            pickle.dump(self.toDict(), f)
    
    def saveB(self, file) -> None:
        """
        The saveB function saves the current state of the object to a file.
            The function takes one argument, which is a string representing the name of
            the file that will be created.  The function then uses pickle to save all
            attributes in self into this new file.
        
        :param self: Represent the instance of the class
        :param file: Specify the file name and location to save the data
        :return: The object that is being saved
        :doc-author: Trelent
        """
        with open(file, "wb") as f:
            pickle.dump(self, f)
        
    @staticmethod
    def load(file) -> "NeuralNetwork":
        """
        The load function takes a file name as an argument and returns a NeuralNetwork object.
        The file must be in the same directory as this script, or you must provide the full path to it.
        
        
        :param file: Specify the file to load
        :return: A neuralnetwork object, so you need to create a new instance of it
        :doc-author: Trelent
        """
        with open(file, "rb") as f:
            datas = pickle.load(f)
        net = NeuralNetwork(0, 0)
        net.weightRange = datas["ranges"]["weight"]
        net.biasRange = datas["ranges"]["bias"]
        net.functions = datas["functions"]
        net.layers = datas["layers"]["lst"]
        net.nbLayer = datas["layers"]["nb"]
        net.weights = [np.array(lst) for lst in datas["weights"]]
        net.biases = [np.array(lst) for lst in datas["biases"]]
        return net
    
    @staticmethod
    def loadB(file) -> "NeuralNetwork":
        """
        The loadB function takes a file name as an argument and returns the contents of that file.
        The function opens the file in binary mode, loads it into memory using pickle, and then closes it.
        
        
        :param file: Load the file that is being passed through
        :return: A network object
        :doc-author: Trelent
        """
        with open(file, "rb") as f:
            net = pickle.load(f)
        return net

