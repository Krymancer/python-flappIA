import numpy as np
from math import exp

from numpy.core.fromnumeric import shape

def sigmod(x):
    return 1 / (1  + exp(-x))

def dsigmod(x):
    return x * (1-x)

vsigmod = np.vectorize(sigmod)
vdsigmod = np.vectorize(dsigmod)

class NeuralNetwork():
    def __init__(self,input_nodes=0,hidden_nodes=0,output_nodes=0,neural_network=None,):
        if neural_network:
            self.input_nodes = neural_network.input_nodes
            self.hidden_nodes = neural_network.hidden_nodes
            self.output_nodes = neural_network.output_nodes

            self.weights_ih = neural_network.weights_ih
            self.weights_ho = neural_network.weights_ho

            self.bias_h = neural_network.bias_h
            self.bias_o = neural_network.bias_o
        else:
            self.input_nodes = input_nodes
            self.hidden_nodes = hidden_nodes
            self.output_nodes = output_nodes

            self.weights_ih = np.random.rand(self.hidden_nodes,self.input_nodes)
            self.weights_ho = np.random.rand(self.output_nodes,self.hidden_nodes)

            self.bias_h = np.random.rand(self.hidden_nodes)
            self.bias_h = self.bias_h[...,None]

            self.bias_o = np.random.rand(self.output_nodes)
            self.bias_o = self.bias_o[...,None]

        self.learning_rate = 0.1

    def predict(self,input_array):
        inputs = np.mat(input_array).transpose()
        hidden = np.matmul(self.weights_ih,inputs)
        hidden = np.add(hidden,self.bias_h)
        hidden = vsigmod(hidden)
        out = np.matmul(self.weights_ho,hidden)
        out = np.add(out,self.bias_o)
        out = vsigmod(out)
        output = []
        for item in range(self.output_nodes):
            output.append(out.item(item))
        return output

    def mutate(self,function):
        vfunction = np.vectorize(function)

        self.weights_ih = vfunction(self.weights_ih)
        self.weights_ho = vfunction(self.weights_ho)
        self.bias_h = vfunction(self.bias_h)
        self.bias_o = vfunction(self.bias_o)

    def get_weights(self):
        return [self.weights_ih, self.weights_ho]

    def set_weights(self,weights):
        self.weights_ih = weights[0]
        self.weigths_ho = weights[1]