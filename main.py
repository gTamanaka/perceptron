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

csv = np.genfromtxt("dataset.csv", delimiter=",")
data_inputs = []
for row in csv:
    data_inputs.append(row)

inputs = data_inputs[0]
print("Tem que ativar")
perceptron.predict(inputs) 

inputs = data_inputs[1]
print("Tem que ativar")
perceptron.predict(inputs)

print("Tem NOT ativar")
inputs = data_inputs[2]
perceptron.predict(inputs) 