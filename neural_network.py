from math import exp
from csv import reader
from random import seed
from random import random
from random import randrange
import matplotlib.pyplot as plt

# load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

# convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

def prepare_data(train_set_name, test_set_name):
    train_set = load_csv(train_set_name)
    test_set = load_csv(test_set_name)

    for i in range(len(train_set[0])-1):
        str_column_to_float(train_set, i)

    for i in range(len(test_set[0])-1):
        str_column_to_float(test_set, i)

    # convert class column to integer
    class_column = len(train_set[0]) - 1
    class_values = [row[class_column] for row in train_set]
    unique = set(class_values)
    lookup = dict()

    for i, value in enumerate(unique):
        lookup[value] = i

    for row in train_set:
        row[class_column] = lookup[row[class_column]]

    for row in test_set:
        row[class_column] = lookup[row[class_column]]

    return (train_set, test_set)


# find the min and max values for each column
def dataset_minmax(dataset):
    minmax = list()
    stats = [[min(column), max(column)] for column in zip(*dataset)]
    return stats

# rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row) - 1):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

# forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row

    for layer in network:
        new_inputs = []

        for neuron in layer:
            # calculate neuron activation for a input
            # activation = sum(weight_i * input_i) + bias
            weights = neuron['weights']
            activation = weights[-1] # bias

            for i in range(len(weights) - 1):
                activation += weights[i] * inputs[i]

            # transfer function
            # sigmoid 1.0 / (1.0 + exp(-activation))
            neuron['output'] = 2.0 * 1.0 / (1.0 + exp(-activation)) - 1.0
            new_inputs.append(neuron['output'])

        inputs = new_inputs

    return inputs

# make a prediction with a network
def predict(network, row):
    outputs = forward_propagate(network, row)

    return outputs.index(max(outputs))

def evaluate_algorithm(train, test, l_rate, e_epoch, n_hidden):
    # backpropagation algorithm with stochatic gradient descent
    predictions = list()
    n_inputs    = len(train[0]) - 1
    n_outputs   = len(set(row[-1] for row in train))
    sse_list    = list()
    
    # initialize a network
    network = list()
    hidden_layer = [ {'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden) ]
    output_layer = [ {'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_outputs) ]
    network.append(hidden_layer)
    network.append(output_layer)

    # train a network for a fixed number of epochs
    for epoch in range(1, n_epoch + 1):
        sum_error = 0

        for row in train:
            outputs = forward_propagate(network, row)

            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            sum_error += sum([(expected[i] - outputs[i])**2 for i in range(len(expected))])

            # backpropagate error and store in neurons
            for i in reversed(range(len(network))):
                layer = network[i]
                errors = list()

                if i != len(network) - 1: # if not output layer
                    for j in range(len(layer)):
                        error = 0.0

                        for neuron in network[i + 1]:
                            error += (neuron['weights'][j] * neuron['delta'])

                        errors.append(error)
                else: # if output layer
                    for j in range(len(layer)):
                        neuron = layer[j]
                        errors.append(expected[j] - neuron['output'])

                for j in range(len(layer)):
                    neuron = layer[j]
                    neuron_output = neuron['output']
                    # transfer derivative
                    # sigmoid: output * (1.0 - output)
                    neuron['delta'] = errors[j] * (0.5 * (1 - neuron_output  * neuron_output))

            # update weights
            # weight = weight + learning_rate * error * input
            for i in range(len(network)):
                inputs = row[:-1]

                if i != 0:
                    inputs = [neuron['output'] for neuron in network[i - 1]]

                for neuron in network[i]:
                    for j in range(len(inputs)):
                        neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]

                    neuron['weights'][-1] += l_rate * neuron['delta']

        sse_list.append(sum_error)

        print('> ep=%3d, lr=%2.3f, err=%.3f' % (epoch, l_rate, sum_error))



    for row in test:
        prediction = predict(network, row)
        predictions.append(prediction)

    actual = [row[-1] for row in test]

    # calculate accuracy percentage
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predictions[i]:
            correct += 1

    return (
        sse_list,
        correct / float(len(actual)) * 100.0
    )




seed(1)

train_set, test_set = prepare_data(
    train_set_name = 'dataset/ionosphere.training_set.data.csv',
    test_set_name = 'dataset/ionosphere.test_set.data.csv'
)

l_rate = 0.1  # 0.1
n_epoch = 400 # 200
n_hidden = 8  # 14

sse_list, scores = evaluate_algorithm(train_set, test_set, l_rate, n_epoch, n_hidden)

print('==============================')
print('CORRECT: %.2f%%' % scores)

plt.plot(range(1, len(sse_list) + 1), sse_list)
plt.ylim([min(sse_list) - 1, max(sse_list) + 1])
plt.ylabel('SSE')
plt.xlabel('Epoki')
plt.tight_layout()
plt.savefig('./figures/cost_%.2f_%d_%d.png' % (l_rate, n_epoch, n_hidden), dpi=300)
plt.show()
