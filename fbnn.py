import numpy as np
from tqdm import tqdm
from tqdm.std import trange
import os

class BPNeuralNetwork():
    def __init__(self, input_n, hidden_n, name):
        np.random.seed(1)
        self.input_n = input_n + 1
        self.hidden_n = hidden_n
        self.name = name
        self.input_weight = 2 * np.random.rand(self.input_n, self.hidden_n) - 1
        self.output_weight = 2 * np.random.rand(self.hidden_n, 1) - 1
        self.learn = 0.05
        if os.path.exists(".\\Checkpoint\\" + self.name + "-input_weight.npy"):
            print("\n\nLoading model...", end = '\n')
            self.load()

    def load(self):
        self.input_weight = np.load(".\\Checkpoint\\" + self.name + "-input_weight.npy")
        self.output_weight = np.load(".\\Checkpoint\\" + self.name + "-output_weight.npy")

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    # def tanh(self, x):
    #     return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    # def tanh_derivative(self, x):
    #     y = self.f(x)
    #     return 1 - np.multiply(y, y)

    def train(self, train_inputs, train_outputs, epochs, value_text, value_label):
        train_iterations = len(train_inputs)
        print("\n\nTrain...")
        f = open(self.name + "_loss.txt", "w")
        for epoch in trange(epochs):
            error = 0.0
            for iteration in range(train_iterations):
                output = self.predict(train_inputs[iteration])

                output_error = train_outputs[iteration] - output
                output_delta = output_error * self.sigmoid_derivative(output)
                output_adjustments = np.dot(self.hidden_cell.reshape(self.hidden_n, 1), output_delta)
                self.output_weight += self.learn * output_adjustments

                hidden_error = np.array([np.dot(output_delta, self.output_weight[i]) for i in range(self.hidden_n)])
                hidden_delta = np.array([self.sigmoid_derivative(self.hidden_cell[i]) * hidden_error[i] for i in range(self.hidden_n)])
                hidden_adjustments = np.dot(self.input_cell.reshape(self.input_n, 1), hidden_delta.reshape(1, self.hidden_n))
                self.input_weight += self.learn * hidden_adjustments

                error += abs(train_outputs[iteration] - output)
            print(f"\nepoch {epoch}: loss {error}", end = '\n')
            f.write(f"\nepoch {epoch}: loss {error}\n")
            if epoch % 10 == 0:
                self.value(value_text, value_label)
        np.save("./Checkpoint/" + self.name + "-input_weight", self.input_weight)
        np.save("./Checkpoint/" + self.name + "-output_weight", self.output_weight)


    def predict(self, inputs):
        self.input_cell = np.zeros(self.input_n)
        self.input_cell[ : self.input_n - 1] = inputs
        self.input_cell[self.input_n - 1] = 1
        self.hidden_cell = np.array([self.sigmoid(np.dot(self.input_cell, self.input_weight[: , j])) for j in range(self.hidden_n)])
        self.output_cell = self.sigmoid(np.dot(self.hidden_cell, self.output_weight[: , 0]))
        return self.output_cell
    
    def value(self, value_text, value_label):
        num = len(value_label)
        tp, fp, fn, tn = 0, 0, 0, 0
        print("\n\nBegin to value...", end = '\n')
        for i in trange(num):
            output = round(self.predict(value_text[i]))
            if output == 0 and value_label[i] == 0:
                tp += 1
            if output == 1 and value_label[i] == 0:
                fn += 1
            if output == 0 and value_label[i] == 1:
                fp += 1
            if output == 1 and value_label[i] == 1:
                tn += 1
        accuracy = (tp + tn) / num
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1score = 2 * (precision * recall) / (precision + recall)
        print(f"Accuracy : {accuracy}; Precision : {precision}; Recall : {recall}; F1score : {f1score}", end = '\n')

