import pandas as pd
import numpy as np

def save_data(data, all_metrics, sampler):
    results_df = pd.DataFrame()

    results_df['Chunk Index'] = data[:,0]

    for m, metric in enumerate(all_metrics):
        results_df[metric.__name__] = data[:,m+1]
        
    results_df.to_csv("results_" + str(sampler) + ".csv", sep=';', index=False)