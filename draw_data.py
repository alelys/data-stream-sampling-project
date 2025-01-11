import matplotlib.pyplot as plt
import numpy as np

def draw(data_array, all_metrics, sampler):

    plt.figure(figsize=(8,4))

    for m, metric in enumerate(all_metrics):
        plt.plot(data_array[:,0], data_array[:,m+1], label=metric.__name__)          

    plt.title(sampler)
    plt.ylim(0, 1.1)
    plt.ylabel('Quality')
    plt.xlabel('Chunk')
    plt.legend()

    plt.show()