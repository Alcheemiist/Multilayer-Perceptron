

import os
import numpy as np
import matplotlib.pyplot as plt

def plot_all_metrics(directory):
    plt.figure(figsize=(12, 4))

    # Plot Losses
    plt.subplot(1, 2, 1)
    for filename in os.listdir(directory):
        if 'loss' in filename:
            # Load the loss data
            losses = np.load(os.path.join(directory, filename))
            epochs = len(losses)  # Derive the number of epochs from the length of the array
            plt.plot(range(epochs), losses, label=filename)
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracies
    plt.subplot(1, 2, 2)
    for filename in os.listdir(directory):
        if 'accuracy' in filename:
            # Load the accuracy data
            accuracies = np.load(os.path.join(directory, filename))
            epochs = len(accuracies)  # Derive the number of epochs from the length of the array
            plt.plot(range(epochs), accuracies, label=filename)
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Usage
directory = '../historics'  
epochs = 501  
plot_all_metrics(directory)
