from activationFunctions import activationFunctions
from random import choice, uniform

class Node:
    def __init__(self, activationFunction, activatoin) -> None:
        self.activationFunction = activationFunction
        self.activation = activatoin
    
    @property
    def activation(self):
        return self.__weight
    @activation.setter
    def activation(self, value):
        if not isinstance(value, float | int):
            raise ValueError("weight must be a int of a float")
        if (value > 1 or value < 0):
            raise ValueError("weight must be between 0 and 1")
        self.__weight = float(value)
    
    @property
    def activationFunction(self):
        return self.__activationFunction
    @activationFunction.setter
    def activationFunction(self, value):
        try:
            if not isinstance(value(0.0), float):
                print(value(0.0))
                ValueError
                raise ValueError("activationFunction must be a function that take a float and return a float")
        except Exception:
            raise ValueError("activationFunction must be a function that take a float and return a float")
        self.__activationFunction = value


    def activate(self, *inputs) -> float:
        return self.activationFunction(sum(inputs) * self.activation)
    
    def randomMutation(self):
        self.activationFunction = choice(activationFunctions)

    def saveString(self) -> str:
        return f"{self.activationFunction.__name__},{self.activation}"
    

    @staticmethod
    def loadString(data : str) -> "Node":
        data = data.split(",")
        if len(data) > 2:
            raise ValueError(f"too many values : {data}")
        if len(data) < 2:
            raise ValueError(f"not enough values : {data}")

        actFunc, weight = data
        for func in activationFunctions:
            if func.__name__ == actFunc: break
        else:
            raise ValueError(f"wrond data for function name : {actFunc}")

        try:
            weight = float(weight)
        except ValueError:
            raise ValueError(f"wrond data for weight : {weight}")
        return Node(func, weight)