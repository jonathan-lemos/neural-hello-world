import numpy as np

np.random.seed(42)

n_inputs = 2
n_outputs = 3

n_points = 1000

# create a vector with n_input rows and n_points columns
sample_data_x = np.random.rand(n_inputs, n_points)

# add rows 0 and 1
sample_data_add = sample_data_x[0, :] + sample_data_x[1, :]
# subtract row 0 - row 1
sample_data_minus = sample_data_x[0, :] - sample_data_x[1, :]
# subtract | row 0 - row 1 |
sample_data_diff = np.abs(sample_data_x[0, :] - sample_data_x[1, :])

# concatenate our 3 outputs
# lists make these 3 rows of n_points columns and not just one array
sample_data_y = np.concatenate(([sample_data_add], [sample_data_diff], [sample_data_minus]))

# train on 80%. validate on 20%
training_ratio = 0.8
n_training_examples = int(n_points * training_ratio)

# get the training ratio split on the data
x_training = sample_data_x[:, :n_training_examples]
x_validation = sample_data_x[:, n_training_examples:]

y_training = sample_data_y[:, :n_training_examples]
y_validation = sample_data_y[:, n_training_examples:]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_inverse(x):
    return np.log(1 / x - 1)


def d_sigmoid(x):
    return np.exp(-x) / (1 + np.exp(-x)) ** 2


x_training_scaled = sigmoid(x_training)
y_training_scaled = sigmoid(y_training)


class layer:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = np.random.randn(output_dim, input_dim)
        self.biases = np.random.randn(output_dim, 1)
        self.activation = np.zeros((output_dim, 1))

    def activate(self, prev_activation):
        vec = prev_activation * self.weights
        next_activation = np.sum(vec, axis=1).reshape(self.output_dim, 1)
        self.activation = sigmoid(next_activation + self.biases)


neural_net = [layer(n_inputs, 16), layer(16, 12), layer(12, 8), layer(8, n_outputs)]


def forward_prop(input_vec):
    neural_net[0].activation = input_vec
    for i in range(1, len(neural_net)):
        neural_net[i].activate(neural_net[i - 1].activation)


def loss(predicted, actual):
    individual_loss = 0.5 * (actual - predicted) ** 2
    return np.mean(individual_loss)


def d_loss(predicted, actual):
    return np.mean(actual - predicted)



print(sample_data_y)

