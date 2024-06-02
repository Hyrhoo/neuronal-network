import math

def indently(x: float) -> float:
    return x

def binaryStep(x: float) -> float:
    return 0.0 if x < 0 else 1

def logistic(x: float) -> float:
    return 1 / (1 + math.exp(-x))

def tanh(x: float) -> float:
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

def smht(x: float) -> float:
    a, b, c, d = 1, 2, 3, 4
    return (math.exp(a*x) - math.exp(-b*x)) / (math.exp(c*x) + math.exp(-d*x))

def relu(x: float) -> float:
    return max(0.0, x)

def gelu(x: float) -> float:
    return (1 / 2) * x * (1 + math.erf(x / math.sqrt(2)))

def softplus(x: float) -> float:
    return math.log(1 + math.exp(x))

def selu(x: float) -> float:
    a = 1.67326
    y = 1.0507
    return y * (a * (math.exp(x) - 1) if x < 0 else x)    

def pRelu(x: float) -> float:
    a = 0.1
    return a * x if x < 0 else x

def silu(x: float) -> float:
    return x / (1 + math.exp(-x))

def gaussian(x: float) -> float:
    return math.exp(-x**2)



activationFunctions = [indently, binaryStep, logistic, tanh, smht, relu, gelu, softplus, selu, pRelu, silu, gaussian]



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from numpy import linspace, average
    for function in  activationFunctions:
        inputs = linspace(-5, 5, 10_000)
        outputs = list(map(function, inputs))
        plt.plot(inputs, outputs)
        plt.title(function.__name__)
        plt.grid()
        print(f"{function.__name__} :\n\t- min : {min(outputs)}\n\t- avg : {average(outputs)}\n\t- max : {max(outputs)}")
        plt.show()
        print("\n\n\n")