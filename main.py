import numpy as np
from perceptron import Perceptron

print("Running Perceptron")
training_inputs = []
csv = np.genfromtxt("models.csv", delimiter=",")
for row in csv:
    training_inputs.append(row)

print(training_inputs)


labels = np.array([1, 0, 0, 0])

perceptron = Perceptron(64)
perceptron.train(training_inputs, labels)

print("Fornecendo testes")

inputs = training_inputs[0]
print("Tem que ativar", inputs)
perceptron.predict(inputs) 

print("Tem NOT ativar", inputs)
inputs = training_inputs[1]
perceptron.predict(inputs) 