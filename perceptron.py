import numpy as np

class Perceptron(object):

    def __init__(self, no_of_inputs, threshold=100, learning_rate=0.01):
        self.threshold = threshold
        self.learning_rate = learning_rate
        # self.weights = np.zeros(no_of_inputs + 1)
        self.weights = np.zeros(no_of_inputs )
           
    def predict(self, inputs):
        # summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        summation = np.dot(inputs, self.weights) 
        if summation > 0:
          activation = 1
          print("Perceptron ativou")
        else:
          activation = -1            
          print("Perceptron nao ativou")
        return activation

    def train(self, training_inputs, labels):
        print("Iniciando treinamento")
        for _ in range(self.threshold):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights += self.learning_rate * (label - prediction) * inputs
                # self.weights[0] += self.learning_rate * (label - prediction)
        print("Terminando treinamento")
        