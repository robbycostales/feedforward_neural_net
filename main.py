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
    def __init__ (self, dimensions, activation_type):
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

        # CREATES NODE LIST
        self.node_list = []
        # creates input nodes
        self.num_inputs = len(dimensions[0])

        # creates hidden layer nodes
        self.num_hidden_layers = len(dimensions) - 2

        # creates output nodes
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
