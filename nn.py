import numpy as np
from math import exp

def sigmod(x):
    return 1 / (1  + exp(-x))

def dsigmod(x):
    return x * (1-x)

class NeuralNetwork():
    def __init__(self,neural_network=None,input_nodes=0,hidden_nodes=0,output_nodes=0):
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
            self.bias_o = np.random.rand(self.output_nodes)

        self.learning_rate = 0.1

    def predict(self,input_array):
        inputs = np.array(input_array)
        hidden = np.matmul(self.weights_ih,inputs)
        hidden = np.add(hidden,self.bias_h)
        hidden = np.array([sigmod(x) for x in hidden])

        out = np.matmul(self.weights_ho,hidden)
        out = np.add(out,self.bias_o)
        out = np.array([sigmod(x) for x in hidden])
        return out

    def copy(self):
        return self

    def mutate(self,function):
        weights_ih = np.array([function(x) for x in self.weights_ih])
        weights_ho = np.array([function(x) for x in self.weights_ho])
        bias_h = np.array([function(x) for x in self.bias_h])
        bias_o = np.array([function(x) for x in self.bias_o])
        self.weights_ih = weights_ih
        self.weights_ho = weights_ho
        self.bias_h = bias_h
        self.bias_o = bias_o


    def train(self,input_array,target_array):
        inputs = np.array(input_array)

        hidden = np.matmul(self.weights_ih,inputs)
        hidden = np.add(hidden,self.bias_h)
        hidden = np.array([sigmod(x) for x in hidden])

        out = np.matmul(self.weights_ho,hidden)
        out = np.add(out,self.bias_h)
        out = np.array([sigmod(x) for x in hidden]) 

        targets = np.array(target_array)

        out_errors = np.subtract(targets,out)
        gradients = np.array([dsigmod(x) for x in out])
        gradients = np.matmul(gradients,out_errors)
        gradients = gradients * self.learning_rate

        hidden_T = np.transpose(hidden)
        weight_ho_deltas = np.matmul(gradients,hidden_T)

        self.weigths_ho = np.add(self.weigths_ho,weight_ho_deltas)
        self.bias_h = np.add(self.bias_h,gradients)

        who_t = np.transpose(self.weigths_ho)
        hidden_errors = np.matmul(who_t,out_errors)
        hidden_gradient = np.array([dsigmod(x) for x in hidden])
        hidden_gradient = np.matmul(hidden_gradient,hidden_errors)
        hidden_gradient = hidden_gradient * self.learning_rate

        inputs_T = np.transpose(inputs)
        weight_ih_deltas = np.matmul(hidden_gradient,inputs_T)
        self.weights_ih = np.add(self.weights_ih,weight_ih_deltas)

        self.bias_h = np.add(self.bias_h,hidden_gradient)