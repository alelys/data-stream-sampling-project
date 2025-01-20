import matplotlib.pyplot as plt
import datetime

def draw(data_array, metrics, samplers, classifier_name, stream_name):
    
    fig, axs = plt.subplots(len(metrics), sharex=True, sharey=True, figsize=(10,7))

    plt.xlabel('Chunk')
    plt.ylabel('Quality')
    fig.suptitle(classifier_name + ", " + stream_name)

    for i, metric in enumerate(metrics):
        for n, sampler in enumerate(samplers):
            axs[i].plot(data_array[data_array[:, 0] == n, 1], data_array[data_array[:, 0] == n, i+2], label=sampler)
            #print(data_array[data_array[:, 0] == n, 1])
            #print(data_array[data_array[:, 0] == n, i+2])
            #print('-------------------------------------------------')
        axs[i].set_title(metric.__name__)
        axs[i].set_yticks([0.25, 0.5, 0.75, 1])      
        axs[i].set_ylim(0, 1.1)        
        axs[i].legend()                  
        #axs[i].grid()



    plt.savefig('ResultsFigs\\fig_' + classifier_name + "_" + stream_name + str(datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")) + ".png", dpi=200)

    plt.show()

