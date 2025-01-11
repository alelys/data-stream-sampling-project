from sklearn.naive_bayes import GaussianNB
from strlearn.streams import StreamGenerator
from sklearn.metrics import accuracy_score, recall_score, precision_score
from imblearn.over_sampling import RandomOverSampler,  SMOTE
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids
from data_processing import process_data
from draw_data import draw
from save_data import save_data



# defining classifier, samplers, metrics and the stream generator
classifier = GaussianNB()
sampler_list = [RandomOverSampler(), SMOTE(), RandomUnderSampler(), ClusterCentroids()]
metrics = [accuracy_score, recall_score, precision_score]
stream_gen = StreamGenerator(n_drifts=3,
                            random_state=2222,
                            n_chunks=40,
                            concept_sigmoid_spacing=10,
                            weights=[0.9, 0.1])        

# choose a sampler to test
used_sampler = sampler_list[0]                                                                          

# process data and get results
results = process_data(clf=classifier, sampler=used_sampler, all_metrics=metrics, stream=stream_gen)   

# visualize data
draw(data_array=results, all_metrics=metrics, sampler=used_sampler)        

# save data to file
save_data(data=results, all_metrics=metrics, sampler=used_sampler)