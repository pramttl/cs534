
# coding: utf-8

# ##### Figuring out how to improve naive bayes classifier
# Design and test a heuristic to reduce the vocabulary size and improve the classification
# performance. This is intended to be open-ended exploration. Please describe clearly what is your strategy
# for reducing the vocabulary size and the results of your exploration. A basic pointer to seed your exploration
# is that we would like to remove words of no discriminative power. How can we measure the discriminative
# power of a word?


import math
import numpy as np
import pandas as pd


labels = pd.read_csv('newsgrouplabels.txt')
labels = np.array(labels['labels'])

train_data = pd.read_csv('train.data', delimiter=' ')
doc_labels = pd.read_csv('train.label', delimiter=' ')

VOCAB_SIZE = 61188

print "====== labels ======"
print labels
print
print "==== train_data ===="
print train_data[0:5]
print
print "==== doc_labels ===="
print doc_labels[0:5]


grouped = doc_labels.groupby(['label',])
label_to_ndocs = grouped.apply(lambda x: len(x))  # Shows number of documents of each label
label_to_ndocs.name = 'doccount'


priors = label_to_ndocs / sum(label_to_ndocs)
priors.name = 'priors'
log_priors = priors.apply(math.log)
log_priors.name = 'log_priors'
label_priors = pd.concat([label_to_ndocs, priors, log_priors], axis=1)
print label_priors


def label_from_docid(docid):
    # doc_labels series is indexed from 0, however docid starts from 1
    return doc_labels['label'][docid-1]


train_data_labels = train_data['docid'].apply(label_from_docid)
train_data['doclabel'] = train_data_labels

grouped = train_data.groupby(['doclabel', 'wid'])

# This takes around 5-10 seconds
s = grouped.apply(lambda x: sum(x['count']))
t = s.unstack()

# Appending blank columns corresponding to each wid which was not present in training dataset but is present in vocabulary
# I am using the property of the dataset here that last set of words in vocabulary are not being used in the training data
# This property can be figured out from shape of t
t = pd.concat([t, pd.DataFrame(columns=range(t.shape[1]+1, VOCAB_SIZE+1))])
t.fillna(0, inplace=True)


# This matrix will be filled with log(p(x|y)) eventually.
matrix = t.transpose()

# Need to get total count of words per class.
# Even in same document there can be same words that are repeated
grouped = train_data.groupby(['doclabel'])
wordcount_given_label = grouped.apply(lambda x: len(x['wid']))   # x['wid'] has words correspondong to each group

# Laplace smoothing (Dirchlet)
ALPHA = 0.01
for l in range(1, len(labels)+1):
    # Laplace smoothing, column by column (or in other words label by label)
    matrix[l] = matrix[l].apply(lambda x: math.log((x + ALPHA) / (wordcount_given_label[l] + ALPHA*VOCAB_SIZE)))


std_deviations = matrix.apply(np.std, axis=1)
std_deviations    # This should give an indication of discriminative power of words


import matplotlib.pyplot as plt
pd.options.display.mpl_style = 'default'

graph = std_deviations.plot()
graph.set_xlabel("word_ids")
graph.set_ylabel("discriminative power")


DISCRIMINATIVE_THRESHOLD = 0.7
mean_discriminative_power = np.mean(std_deviations)
is_word_discriminative = (std_deviations > DISCRIMINATIVE_THRESHOLD)  
                                              # A series which contains true corresponding to words which we consider
                                              # as discriminative
print sum(is_word_discriminative)       # Total number of discriminative words
is_word_discriminative[0:50]
std_deviations


# Loading in test data
test_data = pd.read_csv('test.data', delimiter=' ')
test_doc_labels = pd.read_csv('test.label', delimiter=' ')

# 1. For each document in test data we have to predict it's class
# 2. Algorithm: Add up probabilities corresponding to those words which exisit in the document. This gives log(PROD(p(x|y)). This has to be multipled by p(y) for that document class.

test_docids = pd.unique(test_data.docid.ravel())

# label_pred_matrix
labels = np.array(range(1,21))
label_pred_matrix = pd.DataFrame(0, index=test_docids, columns=labels)
label_pred_matrix    # docids are rows, labels are columns and we want to fill in this matrix


# tdata is numpy array, label_pred_matrix, matrix are data frames
for row_id in test_data.index:
    docid, wid, count = test_data.loc[row_id]

    if docid%100 == 0:
        print docid
    
    # Only consider probabilities of discriminative words, ignore rest
    if is_word_discriminative[wid]:
        label_pred_matrix.loc[docid] += matrix.loc[wid]
    # For each label assumption calculate p(x|y) and fill in prediction matrix
    #for label_assumption in range(1,21):
        #label_pred_matrix[label_assumption][docid] += matrix[label_assumption][wid]


# Adding log(p(y)) to the prediciton matrix
for l in range(1,21):
    label_pred_matrix[l] += label_priors['log_priors'][l]


predictions = label_pred_matrix.apply(np.argmax, axis=1)
predictions.to_csv('predictions_naive_bayes_multinomial_3.txt')


# Accuracy and Confusion matrix code is applied after this



