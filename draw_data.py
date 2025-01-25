import matplotlib.pyplot as plt
import datetime

def draw(data_array, metrics, samplers, classifier_name, stream_name):
    
    sampler_colors = {
        'RandomOverSampler()': 'limegreen',
        'SMOTE()': 'lime',
        'RandomUnderSampler()': 'deepskyblue',
        'ClusterCentroids()': 'steelblue',
        'RandomChoice': 'violet',
    }


    fig, axs = plt.subplots(len(metrics), sharex=True, sharey=True, figsize=(10,7))
    
    plt.xlabel('Chunk')
    fig.suptitle(classifier_name + ", " + stream_name)

    for i, metric in enumerate(metrics):
        for n, sampler in enumerate(samplers):
            color = sampler_colors.get(str(sampler))
            axs[i].plot(data_array[data_array[:, 0] == n, 1], data_array[data_array[:, 0] == n, i+2], label=sampler, color=color)

        axs[i].set_title(metric.__name__)
        axs[i].set_yticks([0.25, 0.5, 0.75, 1])      
        axs[i].set_ylabel('Quality')

        if stream_name == "stream2" or classifier_name == "SGDClassifier":
            axs[i].set_ylim(0.5, 1.05) 
        else:
            axs[i].set_ylim(0, 1.05)


        axs[i].legend(loc="lower left", prop={'size': 8})                  


    plt.savefig('ResultsFigs\\fig_' + classifier_name + "_" + stream_name + str(datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")) + ".png", dpi=200)

    plt.show()

