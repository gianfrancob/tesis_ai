# from tensorflow.python.summary.summary_iterator import summary_iterator
# import sys

# def main(argv):
#     path_to_file = argv[0]

#     for e in summary_iterator(path_to_file):
#         for v in e.summary.value:
#             if v.tag == 'loss' or v.tag == 'accuracy':
#                 print(v.simple_value)

# if __name__ == "__main__":
#     main(sys.argv[1:])

import sys


import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_tensorflow_log(path):

    # Loading too much data is slow...
    tf_size_guidance = {
        'compressedHistograms': 10,
        'images': 0,
        'scalars': 100,
        'histograms': 1
    }

    event_acc = EventAccumulator(path, tf_size_guidance)
    event_acc.Reload()

    # Show all tags in the log file
    print(event_acc.Tags())

    training_loss = event_acc.Scalars('epoch_loss')
    # validation_loss = event_acc_val.Scalars('epoch_loss') # TODO: poner validation loss

    # training_accuracies =   event_acc.Scalars('training-accuracy')
    # validation_accuracies = event_acc.Scalars('validation_accuracy')

    steps = 10
    x = np.arange(steps)
    y = np.zeros([steps, 2])

    # for i in xrange(steps):
    for i in range(steps):
        y[i, 0] = training_loss[i][2]
        y[i, 1] = training_loss[i][2] # TODO: poner validation loss
        # y[i, 0] = training_accuracies[i][2] # value
        # y[i, 1] = validation_accuracies[i][2]

    plt.plot(x, y[:,0], label='training_loss')
    plt.plot(x, y[:,1], label='training_loss') # TODO: poner validation loss
    # plt.plot(x, y[:,0], label='training accuracy')
    # plt.plot(x, y[:,1], label='validation accuracy')

    plt.xlabel("Steps")
    # plt.ylabel("Accuracy")
    plt.ylabel("Loss")
    plt.title("Training Progress")
    plt.legend(loc='upper right', frameon=True)
    plt.show()


if __name__ == '__main__':
    log_file = sys.argv[1:][0]
    plot_tensorflow_log(log_file)