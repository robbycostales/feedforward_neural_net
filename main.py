# Author: Robert Costales
# Date began: 2017-04-14
# Language: Python 3
# File Summary: Main file containing the neural network framework, as well as
#               the specifications of any created neural networks

# IMPORT STATEMENTS ###########################################################

import numpy as np
import copy
import random

# CLASSES #####################################################################


def dot(a, b):
    s = 0
    for i in range(len(a)):
        s += a[i] * b[i]
    return s


def step(a):
    if a < 0:
        return -1
    elif 0 <= a < 0.5:
        return 0
    elif 0.5 <= a <= 1:
        return 1
    else:
        return -1


def sigmoid(inp):
    """
    sigmoid function

    Args:
        inp: from signal
    Returns:
        sigmoid of input
    """
    return 1 / (1 + np.exp(-inp))


class Network:
    def __init__(self, dimensions, learning_rate=1.0, activation_type="adaline"):
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
        # creates bias values
        self.bias = 1
        # learning rate
        self.gamma = learning_rate

        # CREATES NODE LIST
        self.nodes = []

        # initializes input nodes
        self.num_inputs = self.dimensions[0]
        inputs = []
        for i in range(self.num_inputs):
            inputs.append(Node(activation_type=self.avn_type,
                               layer_type="input"))
        self.nodes.append(inputs)

        # initializes hidden layer nodes
        self.num_hidden_layers = len(self.dimensions) - 2
        for i in range(self.num_hidden_layers):
            layer = []
            for j in range(self.dimensions[i+1]):
                layer.append(Node(activation_type=self.avn_type,
                                  layer_type="hidden"))
            self.nodes.append(layer)

        # initializes output nodes
        self.num_outputs = self.dimensions[-1]
        outputs = []
        for i in range(self.num_outputs):
            outputs.append(Node(activation_type=self.avn_type,
                                layer_type="output"))
        self.nodes.append(outputs)

        # CREATES WEIGHT LIST
        # weights[layer][node][specific weights corresponding to bias +
        # previous nodes]
        self.weights = []
        # value all weights will be initialized to
        weight_init_val = 0.5

        # inputs do not have weights (to left), so we use empty sub-lists
        input_weights = []
        for i in range(self.num_inputs):
            input_weights.append([])
        self.weights.append(input_weights)

        # all net weights
        for layer in range(1, self.num_hidden_layers + 2):
            layer_weights = []
            for node in range(len(self.nodes[layer])):
                node_weights = []
                # take all previous nodes into account +1 for the bias
                for i in range(len(self.nodes[layer-1])+1):
                    node_weights.append(copy.deepcopy(random.random()))
                # some for loop here to add to node_weights
                layer_weights.append(copy.deepcopy(node_weights))
            self.weights.append(copy.deepcopy(layer_weights))

    def __str__(self):
        return "include specs here (will update later)"

    def feedforward(self, inputs):
        """
        Feed-forward of neural network

        Returns:
            outputs (in a list)
        """
        # number of inputs should match number of input nodes
        if len(inputs) != self.num_inputs:
            # raise "error"
            print("feed-forward: error regarding number of inputs")
            return []
        else:
            # set all outputs of input nodes to "inputs" list
            for i in range(self.num_inputs):
                self.nodes[0][i].output = inputs[i]

        # find all other nodes' outputs
        for layer in range(1, len(self.nodes)):
            for node in range(len(self.nodes[layer])):
                weights = self.weights[layer][node][:]
                values = [j.output for j in self.nodes[layer-1]]
                values += [self.bias]
                self.nodes[layer][node].output = sigmoid(dot(weights, values))

        outputs = [j.output for j in self.nodes[-1]]
        # print(outputs)
        return outputs

    def backprop(self, t):
        """
        Updating weights of neural network through back-propagation

        Returns:
            void
        """

        # DELTA RULE
        # paths in form: (path value, origin node)
        # origin node used for weight indexing for next layer

        # only one "old" path at start
        old_paths = [1]
        # to fill during next loop
        new_paths = []

        # for each node in the last layer
        for node in range(len(self.weights[-1])):
            for path in old_paths:
                # for each weight connecting the nodes in the previous layer to
                # the current node
                out_pi = self.nodes[-1][node].output
                new_paths.append((path * (out_pi - t[node]) * out_pi * (1 - out_pi), node))
                for weight in range(len(self.weights[-1][node])):
                    # if it is the last weight, it corresponds to the bias
                    if weight == len(self.weights[-1][node]) - 1:
                        out_xi = self.bias
                    else:
                        out_xi = self.nodes[-2][weight].output

                    # we find out delta
                    dw = path*(out_pi - t[node]) * out_pi * (1 - out_pi) * out_xi
                    # update the weights
                    self.weights[-1][node][weight] -= self.gamma * dw

        # all other weights
        for layer in range(len(self.weights)-2, 0, -1):
            old_paths = []
            old_paths += new_paths
            new_paths = []
            for node in range(len(self.nodes[layer])):
                for path in old_paths:
                    out_p = self.nodes[layer][node].output
                    w_p = self.weights[layer+1][path[1]][node]

                    # create new path for each new node * old_path
                    new_paths.append((path[0] * w_p * out_p * (1 - out_p), node))

                    # for each weight connecting the nodes in the previous
                    # layer to the current node
                    for weight in range(len(self.weights[layer][node])):
                        # if it is the last weight, it corresponds to the bias
                        if weight == len(self.weights[layer][node]) - 1:
                            # equals bias on left
                            out_x = self.bias
                        else:
                            # equals node on left that weight is connected to
                            out_x = self.nodes[layer-1][weight].output

                        # we find out delta
                        dw = path[0] * w_p * out_p * (1-out_p) * out_x
                        # update the weights
                        self.weights[layer][node][weight] -= self.gamma * dw

    def training(self, iterations, inputs, targets):
        """
        Feeds inputs forward, compares to targets, uses back-propagation to
        update weights
        Supervised learning

        Args:
            iterations : number of training iterations
            inputs : data input values
            targets: corresponding target values

        Returns:
            void
        """
        # print(self.weights)
        for i in range(iterations):
            for n in range(len(inputs)):
                # updates output values of each node
                self.feedforward(inputs[n])

                # updates weights - uses back-propagation
                self.backprop(targets[n])
                # print(self.weights)

    def testing(self, inputs, targets):
        """
        Prints all sorts of fun stuff

        :param inputs:
        :param targets:
        :return: void
        """

        # title
        for j in range(len(targets[0])):
            node_number = "N#"
            raw = "Raw"
            rounded = "Rnd"
            target = "Tgt"
            status = "Sts"
            print("   {0:<4}{1:<8}{2:<4}{3:<4}{4:<6}".format(node_number, raw, rounded, target, status), end="    ")
        print("\n")

        # actual input results
        for i in range(len(targets)):
            self.feedforward(inputs[i])
            for j in range(self.num_outputs):
                node_number = j
                raw = "{0:.3f}".format(self.nodes[-1][j].output)
                target = targets[i][j]
                rounded = step(float(raw))
                if rounded == target:
                    status = "{0}".format(True)
                else:
                    status = "{0}".format(False)
                print("   {0:<4}{1:<8}{2:<4}{3:<4}{4:<6}".format(node_number, raw, rounded, target, status), end="    ")
            print(" ")
        print(" ")

class Node:
    def __init__(self, activation_type, layer_type):
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
        # output after activation function
        self.output = 0
        self.layer_type = layer_type


# Initialize data sets ########################################################

bin_set = [[0, 0],
           [0, 1],
           [1, 0],
           [1, 1]]

and_target = [[0],
              [0],
              [0],
              [1]]

xor_target = [[0],
              [1],
              [1],
              [0]]

xorplus_target = [[0, 1],
                  [1, 0],
                  [1, 0],
                  [0, 1]]

andplus_target = [[0, 1],
                  [0, 1],
                  [0, 1],
                  [1, 0]]


# print("\n3 Layer 'XOR' and '!XOR' Check:\n")
#
# xor_net = Network([2, 4, 2], learning_rate=.3)
#
# xor_net.training(4000, bin_set, xorplus_target)
# xor_net.testing(inputs=bin_set, targets=xorplus_target)


print("\n3 Layer 'AND' and '!AND' Check:\n")

xor_net = Network([2, 4, 2], learning_rate=4)

xor_net.training(5000, bin_set, plus_target)
xor_net.testing(inputs=bin_set, targets=andplus_target)
