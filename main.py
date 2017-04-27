# Author: Robert Costales
# Date began: 2017-04-14
# Language: Python 3
# File Summary: Main file containing the neural network framework, as well as
#               the specifications of any created neural networks

# _-_-_-_-_-_-_-_-_-_-_-_-_-_-_ IMPORT STATEMENTS _-_-_-_-_-_-_-_-_-_-_-_-_-_ #

import numpy as np
import math
import random as r
import copy as c

# _-_-_-_-_-_-_-_-_-_-_-_-_-_-_ CLASSES _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_ #

class Network:
    def __init__ (self, dimensions, activation_type="adaline"):
        """
        Initializes network
        1. Creates nodes based on activation_type
        2. Stores all of the node objects in a list

        Args:
            dimensions: List of how many nodes in each layer. Includes input
            nodes and output nodes.
            activation_type: what kind of activation function will be used
            most often. Can be changed for individual nodes
        Returns:
            void.
        """
        # creates class attribute of dimensions
        self.dimensions = dimensions
        self.avn_type = activation_type

        # CREATES NODE LIST
        self.node_list = []

        # initializes input nodes
        self.num_inputs = len(dimensions[0])
        inputs = []
        for i in range(self.num_inputs):
            inputs.append(Node(self.avn_type, "input", [], []))
        self.node_list.append(inputs)

        # initializes hidden layer nodes
        self.num_hidden_layers = len(self.dimensions) - 2
        for i in range(self.num_hidden_layers):
            layer = []
            for j in range(self.dimensions[i+1]):
                layer.append(Node(self.avn_type, "hidden", [], [])) ###########

        # initializes output nodes
        self.num_outputs = len(dimensions[-1])

    def feedforward (self):
        """
        Feedforward of neural network

        Returns:
            outputs (in a list)
        """

    def backprop (self):
        """
        Updating weights of neural network through backpropagation

        Returns:
            void

        """

class Node:
    def __init__ (self, activation_type, layer_type, prev_nodes, next_nodes):
        """
        Initializes Node

        Args:
            activation_type: type of activation function used
            layer_type: input, hidden, or output
            prev_nodes: previous nodes to match with weights (includes a "1"
            for the bias)
            next_nodes: next nodes (includes a "1" for the bias)

        """
        self.activation_type = activation_type
        self.layer_type = layer_type
        self.prev_nodes = prev_nodes
        # initializes random weights corresponding to each input node
        self.weights = [random.random() for i in range(len(prev_nodes))]
        self.next_nodes = next_nodes
        self.output = 0

    def get_signal (self):
        """
        obtains signal from previous nodes to be passed through an activation
        function

        Returns:
            dot product of weights and previous node outputs

        """
        return numpy.dot(self.weights, [i.output for i in self.prev_nodes])

    def sigmoid (self, inp):
        """
        sigmoid function

        Args:
            inp: from signal
        Returns:
            sigmoid of input
        """
        return 1/(1 + np.exp(-inp))

    def output (self):
        x = self.get_signal()
        self.output = self.sigmoid(x)
        return self.output


net_a = [2, 4, 1]
