import numpy as np
import random


def process_data(clf, samplers, stream, metrics):

    store_results = []

    for i, sampler in enumerate(samplers):
           
        stream.reset()

        for X_chunk, y_chunk in stream:

            if sampler == "RandomChoice":
                new_sampler = random.choice(samplers[:-1])
                #print("Now using: " + new_sampler.__class__.__name__)
            else:
                new_sampler = sampler
                #print("Now using: " + str(new_sampler))


            X_resampled, y_resampled = new_sampler.fit_resample(X_chunk, y_chunk)

            if stream.chunk_id == 0:
                clf.fit(X_resampled, y_resampled)

            if stream.chunk_id != 0:
                clf.partial_fit(X_resampled, y_resampled)
            
            b_accuracy_calc = metrics[0](y_resampled, clf.predict(X_resampled))
            recall_calc = metrics[1](y_resampled, clf.predict(X_resampled))     
            precision_calc = metrics[2](y_resampled, clf.predict(X_resampled))

            store_results.append([i, stream.chunk_id, b_accuracy_calc, recall_calc, precision_calc])

    return np.array(store_results)
