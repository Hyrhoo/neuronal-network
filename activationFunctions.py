import math

def indently(x: float) -> float:
    return x

def binaryStep(x: float) -> float:
    return 0.0 if x <= 0 else 1

def sigmoid(x: float) -> float:
    try:
        return 1 / (1 + math.exp(-x))
    except:
        return 0

def tanh(x: float) -> float:
    try:
        return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))
    except:
        return 1.0 if x >= 0 else -1.0

def smht(x: float) -> float:
    a, b, c, d = 1, 2, 3, 4
    try:
        return (math.exp(a*x) - math.exp(-b*x)) / (math.exp(c*x) + math.exp(-d*x))
    except:
        return 0

def relu(x: float) -> float:
    return max(0.0, x)

def gelu(x: float) -> float:
    return (1 / 2) * x * (1 + math.erf(x / math.sqrt(2)))

def softplus(x: float) -> float:
    try:
        return math.log(1 + math.exp(x))
    except Exception as e:
        return x

def selu(x: float) -> float:
    a = 1.67326
    y = 1.0507
    try:
        return y * (a * (math.exp(x) - 1) if x < 0 else x)    
    except:
        return y * x

def pRelu(x: float) -> float:
    a = 0.1
    return a * x if x < 0 else x

def silu(x: float) -> float:
    try:
        return x / (1 + math.exp(-x))
    except:
        return x if x >= 0 else 0

def gaussian(x: float) -> float:
    try:
        return math.exp(-x**2)
    except:
        return 0

def sigmoidRelu(x):
    return (sigmoid(relu(x)) - 0.5) * 2

activationFunctions = [indently, binaryStep, sigmoid, tanh, smht, relu, gelu, softplus, selu, pRelu, silu, gaussian, sigmoidRelu]



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from numpy import linspace, average
    for function in  activationFunctions:
        inputs = linspace(-50, 50, 10_000)
        outputs = list(map(function, inputs))
        plt.plot(inputs, outputs)
        plt.title(function.__name__)
        plt.grid()
        print(f"{function.__name__} :\n\t- min : {min(outputs)}\n\t- avg : {average(outputs)}\n\t- max : {max(outputs)}")
        plt.show()
        print("\n\n\n")