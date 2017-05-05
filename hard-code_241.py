# Author: Robert Costales
# Language: Python 3
# Date: 4/3/2017
# Summary: This program is a hard-coded MLP with one hidden layer
#		   The weights are trained on XOR


import random
import math
import copy

# _-_-_-_-_-_-_-_-_-_-_-_ SETTINGS _-_-_-_-_-_-_-_-_-_-_-_ #

NUM_ITER = 10000  # number of times to train through set
GAMMA = 1  # learning rate
FREQ = 20    # frequency of printing values

# _-_-_-_-_-_-_-_-_-_-_-_ WEIGHTS _-_-_-_-_-_-_-_-_-_-_-_ #

# x1, x2 are inputs
# a, b, c, d are hidden layer nodes
# p is output

# weights for a
w_a0 = random.random()  # threshold
w_a1 = random.random()  # for x1 feed
w_a2 = random.random()  # for x2 feed

# weights for b
w_b0 = random.random()  # threshold
w_b1 = random.random()  # for x1 feed
w_b2 = random.random()  # for x2 feed

# weights for c
w_c0 = random.random()  # threshold
w_c1 = random.random()  # for x1 feed
w_c2 = random.random()  # for x2 feed

# weights for d
w_d0 = random.random()  # threshold
w_d1 = random.random()  # for x1 feed
w_d2 = random.random()  # for x2 feed

# weights for p
w_p0 = random.random()  # threshold
w_p1 = random.random()  # for a feed
w_p2 = random.random()  # for b feed
w_p3 = random.random()  # for c feed
w_p4 = random.random()  # for d feed

weights = [[w_p0, w_p1, w_p2, w_p3, w_p4], [w_a0, w_a1, w_a2], [w_b0, w_b1, w_b2], [w_c0, w_c1, w_c2], [w_d0, w_d1, w_d2]]

# _-_-_-_-_-_-_-_-_-_-_-_ TRAINING  _-_-_-_-_-_-_-_-_-_-_-_ #

# xor training set:
xor_set = [[0, 0, 0],
           [0, 1, 1],
           [1, 0, 1],
           [1, 1, 0]]

for i in range(NUM_ITER):
    local_errors = []
    if i % FREQ == 0:
        prev_weights = copy.deepcopy(weights)
        weights = [[w_p0, w_p1, w_p2, w_p3, w_p4], [w_a0, w_a1, w_a2], [w_b0, w_b1, w_b2], [w_c0, w_c1, w_c2],
                   [w_d0, w_d1, w_d2]]

        print("{:<8.3f}{:<8.3f}{:<8.3f}{:<8.3f}{:<8.3f}{:<8.3f}{:<8.3f}{:<8.3f}{:<8.3f}{:<8.3f}{:<8.3f}{:<8.3f}"
              "{:<8.3f}{:<8.3f}{:<8.3f}{:<8.3f}{:<8.3f}   ".format(*weights[0], *weights[1], *weights[2], *weights[3], *weights[4]), end="")
        del_weights = copy.deepcopy(weights)
        for i in range(len(weights)):
            for j in range(len(weights[i])):
                del_weights[i][j] = weights[i][j] - prev_weights[i][j]
        print("{:<8.3f}{:<8.3f}{:<8.3f}{:<8.3f}{:<8.3f}{:<8.3f}{:<8.3f}{:<8.3f}{:<8.3f}{:<8.3f}{:<8.3f}{:<8.3f}"
              "{:<8.3f}{:<8.3f}{:<8.3f}{:<8.3f}{:<8.3f}  ".format(*del_weights[0], *del_weights[1], *del_weights[2], *del_weights[3],
                                                                  *del_weights[4]))
    for j in xor_set:
        x1 = j[0]       # first input
        x2 = j[1]       # second inputs
        target = j[2]   # expected output (target)

        # FEED FORWARD

        # hidden nodes
        net_a = (w_a0 * 1) + (w_a1 * x1) + (w_a2 * x2)

        out_a = 1 / (1 + math.exp(0 - net_a))

        net_b = (w_b0 * 1) + (w_b1 * x1) + (w_b2 * x2)
        out_b = 1 / (1 + math.exp(0 - net_b))

        net_c = (w_c0 * 1) + (w_c1 * x1) + (w_c2 * x2)
        out_c = 1 / (1 + math.exp(0 - net_c))

        net_d = (w_d0 * 1) + (w_d1 * x1) + (w_d2 * x2)
        out_d = 1 / (1 + math.exp(0 - net_d))

        # p node
        net_p = (w_p0 * 1) + (w_p1 * out_a) + (w_p2 * out_b) + (w_p3 * out_c) + (w_p4 * out_d)
        out_p = 1 / (1 + math.exp(0 - net_p))

        # error
        error = (0.5 * (target - out_p) ** 2)
        local_errors.append(error)


        # DELTA RULE (for output weights)
        # output weights
        del_w_p1 = (out_p - target) * out_p * (1 - out_p) * out_a
        w_p1 -= GAMMA * del_w_p1

        del_w_p2 = (out_p - target) * out_p * (1 - out_p) * out_b
        w_p2 -= GAMMA * del_w_p2

        del_w_p3 = (out_p - target) * out_p * (1 - out_p) * out_c
        w_p3 -= GAMMA * del_w_p3

        del_w_p4 = (out_p - target) * out_p * (1 - out_p) * out_d
        w_p4 -= GAMMA * del_w_p4

        w_p0 -= GAMMA * ((out_p - target) * out_p * (1 - out_p) * 1)

        # HIDDEN LAYER UPDATES
        # a weights

        path = (out_p - target) * out_p * (1 - out_p)
        # sets path value to make code easier to read

        del_w_a0 = path * w_p1 * out_a*(1 - out_a) * 1
        w_a0 -= GAMMA * del_w_a0

        del_w_a1 = path*w_p1 * out_a*(1 - out_a) * x1
        w_a1 -= GAMMA * del_w_a1

        del_w_a2 = path*w_p1 * out_a*(1 - out_a) * x2
        w_a2 -= GAMMA * del_w_a2

        # b weights
        del_w_b0 = path* w_p2 * out_b * (1 - out_b) * 1
        w_b0 -= GAMMA * del_w_b0

        del_w_b1 = path* w_p2 * out_b * (1 - out_b) * x1
        w_b1 -= GAMMA * del_w_b1

        del_w_b2 = path* w_p2 * out_b * (1 - out_b) * x2
        w_b2 -= GAMMA * del_w_b2

        # c weights
        del_w_c0 = path* w_p3 * out_c * (1 - out_c) * 1
        w_c0 -= GAMMA * del_w_c0

        del_w_c1 = path* w_p3 * out_c * (1 - out_c) * x1
        w_c1 -= GAMMA * del_w_c1

        del_w_c2 = (out_p - target) * out_p * (1 - out_p) * w_p3 * out_c * (1 - out_c) * x2
        w_c2 -= GAMMA * del_w_c2

        # d weights
        del_w_d0 = path * w_p4 * out_d * (1 - out_d) * 1
        w_d0 -= GAMMA * del_w_d0

        del_w_d1 = path * w_p4 * out_d * (1 - out_d) * x1
        w_d1 -= GAMMA * del_w_d1

        del_w_d2 = path * w_p4 * out_d * (1 - out_d) * x2
        w_d2 -= GAMMA * del_w_d2


# TESTING
print("\nTESTING\n")

print("{0:<8}{1:<8}{2:<8}{3:<8}{4:<8}\n".format("Inp 1", "Inp 2", "Exp", "Act", "Status", "Num"))
for j in xor_set:
    weights = [[w_p0, w_p1, w_p2, w_p3, w_p4], [w_a0, w_a1, w_a2], [w_b0, w_b1, w_b2], [w_c0, w_c1, w_c2],
               [w_d0, w_d1, w_d2]]
    x1 = j[0]
    x2 = j[1]
    target = j[2]

    # FEED FORWARD

    # hidden nodes
    net_a = (w_a0 * 1) + (w_a1 * x1) + (w_a2 * x2)
    out_a = 1 / (1 + math.exp(0 - net_a))

    net_b = (w_b0 * 1) + (w_b1 * x1) + (w_b2 * x2)
    out_b = 1 / (1 + math.exp(0 - net_b))

    net_c = (w_c0 * 1) + (w_c1 * x1) + (w_c2 * x2)
    out_c = 1 / (1 + math.exp(0 - net_c))

    net_d = (w_d0 * 1) + (w_d1 * x1) + (w_d2 * x2)
    out_d = 1 / (1 + math.exp(0 - net_d))

    # p node
    net_p = (w_p0 * 1) + (w_p1 * out_a) + (w_p2 * out_b) + (w_p3 * out_c) + (w_p4 * out_d)
    out_p = 1 / (1 + math.exp(0 - net_p))

    if round(target, 1) == round(out_p, 1):
        status = "Success"
    else:
        status = "Fail"
    print("{0:<8}{1:<8}{2:<8}{3:<8}{4:<8} {5:<8}".format(x1, x2, target, round(out_p, 2), status, out_p))

print("\nEl Fin")

print([[w_p0, w_p1, w_p2, w_p3, w_p4], [w_a0, w_a1, w_a2], [w_b0, w_b1, w_b2], [w_c0, w_c1, w_c2],
               [w_d0, w_d1, w_d2]])

waits = [[[], []], [[w_a1, w_a2, w_a0], [w_b1, w_b2, w_b0], [w_c1, w_c2, w_c0], [w_d1, w_d2, w_d0]], [[w_p1, w_p2, w_p3, w_p4, w_p0]]]

print("\n\n{0}".format(waits))
