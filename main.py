import numpy as np
import random
from functools import reduce


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_inverse(x):
    return -np.log(1 / x - 1)


def d_sigmoid(x):
    return np.exp(-x) / (1 + np.exp(-x)) ** 2


def cost(output, expected):
    return 0.5 * (output - expected) ** 2


def d_cost(output, expected):
    return output - expected


def mini_batches(training_data, mini_batch_size):
    new_data = random.sample(training_data, len(training_data))
    i = 0
    while i < len(new_data):
        yield training_data[i:i + mini_batch_size]
        i += mini_batch_size


class Network:
    def __init__(self, *layer_sizes):
        self.num_layers = len(layer_sizes)
        self.sizes = layer_sizes
        self.biases = [np.random.randn(y, 1) for y in layer_sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]

    def forward_prop(self, x):
        return reduce(lambda a, c: sigmoid(np.dot(c[1], a) + c[0]), zip(self.biases, self.weights), np.array(x).reshape(len(x), 1))

    def measure_cost(self, t_input, t_output):
        return 2 * np.mean(cost(self.forward_prop(t_input), t_output))

    def gradient_descent(self, training_data, mini_batch_size, n_epochs, learning_rate, test_data=None):
        for epoch in range(n_epochs):
            print(f"Epoch {epoch + 1} of {n_epochs} training")

            for mini_batch in mini_batches(training_data, mini_batch_size):
                total_biases_nudge = [np.zeros(bcol.shape) for bcol in self.biases]
                total_weights_nudge = [np.zeros(wcol.shape) for wcol in self.weights]

                for mb_input, mb_output in mini_batch:
                    nudge_biases, nudge_weights = self.back_prop(mb_input, mb_output)
                    total_biases_nudge = [n + t for n, t in zip(nudge_biases, total_biases_nudge)]
                    total_weights_nudge = [n + t for n, t in zip(nudge_weights, total_weights_nudge)]

                self.biases = [bcol - nbcol * learning_rate / len(mini_batch)
                                for bcol, nbcol in zip(self.biases, total_biases_nudge)]
                self.weights = [wcol - nwcol * learning_rate / len(mini_batch)
                                for wcol, nwcol in zip(self.weights, total_weights_nudge)]

            if test_data is None:
                continue

            print(f"Cost: {np.mean([self.measure_cost(t_input, t_output) for t_input, t_output in test_data])}")

    def back_prop(self, input, output):
        nudge_biases = [None for _ in self.biases]
        nudge_weights = [None for _ in self.weights]

        zs = []
        activations = [np.array(input).reshape(len(input), 1)]
        cur_activation = activations[0]

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, cur_activation) + b
            a = sigmoid(z)
            zs.append(z)
            activations.append(a)
            cur_activation = a

        delta = d_cost(activations[-1], np.array(output).reshape(len(output), 1)) * d_sigmoid(zs[-1])
        nudge_biases[-1] = delta
        nudge_weights[-1] = np.dot(delta, activations[-2].transpose())

        for i in range(2, self.num_layers):
            delta = np.dot(self.weights[-i + 1].transpose(), delta) * d_sigmoid(zs[-i])
            nudge_biases[-i] = delta
            nudge_weights[-i] = np.dot(delta, activations[-i - 1].transpose())

        return nudge_biases, nudge_weights


random.seed(0)


def rand():
    return random.gauss(0, 1)


n_points = 100000
data_x = [(rand(), rand()) for _ in range(n_points)]
# data_y = [(sigmoid(a + b), sigmoid(a * b), sigmoid(a - b)) for a, b in data_x]
data_y = [(0.25, -0.25, 0) for a, b in data_x]
data = list(zip(data_x, data_y))

training_ratio = 0.8
n_training = int(n_points * training_ratio)
data_training = data[:n_training]
data_test = data[n_training:]

net = Network(2, 16, 12, 3)
net.gradient_descent(data_training, n_training // 100, 5, 0.05, data_test)

res = net.forward_prop((0, 0))
print(res)

res = net.forward_prop((-0.25, 0.5))
print(res)

res = net.forward_prop((0.5, -0.25))
print(res)

res = net.forward_prop((0.25, 0.5))
print(res)

res = net.forward_prop((0.5, -0.75))
print(res)
