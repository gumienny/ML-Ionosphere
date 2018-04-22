import matplotlib.pyplot as plt
from csv import reader
from math import exp
import random

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

    return outputs.index(max(outputs)), outputs

def train_network(train, test, l_rate, e_epoch, n_hidden, lr_dec = 0.7, lr_inc = 1.05, er = 1.04):
    # backpropagation algorithm with stochatic gradient descent
    n_inputs    = len(train[0]) - 1
    n_outputs   = len(set(row[-1] for row in train))
    sse_list    = list()
    prev_error = 0
    
    # initialize a network
    network = list()

    hidden_layer = [{'weights': [random.random() for i in range(n_inputs + 1)]} for i in range(n_hidden) ]
    output_layer = [{'weights': [random.random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]

    network.append(hidden_layer)
    network.append(output_layer)

    # train a network for a fixed number of epochs
    for epoch in range(1, n_epoch + 1):
        sum_error = 0

        if RANDOMIZE:
            random.shuffle(train)

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

        if ADAPTIVE_LEARNING_RATE:
            if (epoch == 1):
                prev_error = sum_error

            if sum_error > (er * prev_error):
                l_rate *= lr_dec
            elif sum_error < prev_error:
                l_rate *= lr_inc

            prev_error = sum_error

        if VERBOSE:
            print('> epoch={:3d}, lr={:.3f}, sse={:.3f}'.format(epoch, l_rate, sum_error))


    # calculate accuracy percentage
    correct = 0
    
    for i, row in enumerate(test):
        predicted, outputs = predict(network, row)

        if row[-1] == predicted:
            correct += 1

        if VERBOSE:
            if (OUTPUT_ONLY_BAD_PREDICTIONS and (row[-1] != predicted)) or (not OUTPUT_ONLY_BAD_PREDICTIONS):
                answer = 'good' if predicted == row[-1] else 'bad'
                print("%4s - [%s]" % (answer, ', '.join('{:.2f}'.format(i) for i in outputs)))

    return sse_list, correct / float(len(test)) * 100.0


random.seed(7)


train_set, test_set = prepare_data(
    train_set_name = 'dataset/ionosphere.training_set.data.csv',
    test_set_name = 'dataset/ionosphere.test_set.data.csv'
)


l_rate = 0.5 # 0.1
n_epoch = 100 # 200
n_hidden = 10 # 14


OUTPUT_ONLY_BAD_PREDICTIONS = True
ADAPTIVE_LEARNING_RATE = False
RANDOMIZE = False
VERBOSE = True


if False:
    print('[')
    for hidden_neurons in [3, 5, 8, 10, 15]:
        for learning_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            sse_list, scores = train_network(train_set, test_set, learning_rate, n_epoch, hidden_neurons)
            print('[%d, %f, %.2f],' % (hidden_neurons, learning_rate, scores))
    print(']')


if True:
    sse_list, scores = train_network(train_set, test_set, l_rate, n_epoch, n_hidden)
    print('==============================')
    print('CORRECT: {:.2f}%'.format(scores))

    plt.plot(range(1, len(sse_list) + 1), sse_list)
    plt.ylim([min(sse_list) - 1, max(sse_list) + 1])
    plt.ylabel('SSE')
    plt.xlabel('Epoki')
    plt.tight_layout()
    plt.savefig('./figures/cost_{:.2f}_{:d}_{:d}.png'.format(l_rate, n_epoch, n_hidden), dpi=300)
    plt.show()
