import numpy as np

def process_data(clf, sampler, stream, all_metrics):
    store_results = []

    for X_chunk, y_chunk in stream:
        X_resampled, y_resampled = sampler.fit_resample(X_chunk, y_chunk)

        if stream.chunk_id == 0:
            clf.fit(X_resampled, y_resampled)

        if stream.chunk_id != 0:
            clf.partial_fit(X_resampled, y_resampled)
        
        accuracy_calc = all_metrics[0](y_resampled, clf.predict(X_resampled))
        recall_calc = all_metrics[1](y_resampled, clf.predict(X_resampled))     
        precision_calc = all_metrics[2](y_resampled, clf.predict(X_resampled))

        store_results.append([stream.chunk_id, accuracy_calc, recall_calc, precision_calc])

    return np.array(store_results)