from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score
from imblearn.over_sampling import RandomOverSampler,  SMOTE
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids
import data_processing
from draw_data import draw
import streams


# defining classifiers, samplers, metrics and the stream generator
classifiers = [GaussianNB(), MLPClassifier(max_iter=500)]
sampler_list = [RandomOverSampler(), SMOTE(), RandomUnderSampler(), ClusterCentroids(), "RandomChoice"]
metrics_list = [balanced_accuracy_score, recall_score, precision_score]


# choose a sampler, classifier and stream for testingk
used_classifier = classifiers[0]                                                                  
used_stream = streams.stream0


# process data and get results
results = data_processing.process_data(
    clf=used_classifier,
    metrics=metrics_list,
    stream=used_stream,
    samplers=sampler_list)     
print(results)

# visualize data
draw(data_array=results,
     metrics=metrics_list,
     samplers=sampler_list,
     classifier_name=used_classifier.__class__.__name__,
     stream_name=used_stream.name)        




#The balanced accuracy in binary and multiclass classification problems to deal with imbalanced datasets. It is defined as the average of recall obtained on each class.
# The best value is 1 and the worst value is 0

# The recall is intuitively the ability of the classifier to find all the positive samples.
# The best value is 1 and the worst value is 0.

# The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.
# The best value is 1 and the worst value is 0.