
# coding: utf-8

import math
import numpy as np

# We use pandas for dealing with dataframes which are like matrices but allow
# adding labels making inspection easier and alllow us to perform group/aggregate
# operations which are a covenient alternative for writing loops
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


#print doc_labels
grouped = doc_labels.groupby(['label',])
#print grouped.groups
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

def doccount_from_label(label):
    """
    Returns count of number of documents of given label
    """
    # label priors is indexed from 1
    return label_priors['doccount'][label]


train_data_labels = train_data['docid'].apply(label_from_docid)

# Adding label columm to training data frame which tells label
# alongside docid and wid
train_data['doclabel'] = train_data_labels

#train_data

grouped = train_data.groupby(['doclabel', 'wid'])
#grouped.groups

# This takes around 5-10 seconds
s = grouped.apply(lambda x: sum(x['count']))
s

t = s.unstack()

# Appending blank columns corresponding to each wid which was not present in training dataset but is present in vocabulary
# I am using the property of the dataset here that last set of words in vocabulary are not being used in the training data
# This property can be figured out from shape of t
t = pd.concat([t, pd.DataFrame(columns=range(t.shape[1]+1, VOCAB_SIZE+1))])
t.fillna(0, inplace=True)

matrix = t.transpose()
matrix.shape  # (61188, 20) which is VOCAB_SIZE X len(labels)
#matrix

# Need to get total count of words per class.
# Even in same document there can be same words that are repeated
grouped = train_data.groupby(['doclabel'])
wordcount_given_label = grouped.apply(lambda x: len(x['wid']))   # x['wid'] has words correspondong to each group

# Laplace smoothing (Dirchlet alpha can be set)
ALPHA = 0.1
for l in range(1, len(labels)+1):
    # Laplace smoothing, column by column (or in other words label by label)
    matrix[l] = matrix[l].apply(lambda x: math.log((x + ALPHA) / (wordcount_given_label[l] + ALPHA*VOCAB_SIZE)))


# Matrix now has log probabilities
print "==== Smoothing completed ===="


# Loading test data
test_data = pd.read_csv('test.data', delimiter=' ')
test_doc_labels = pd.read_csv('test.label', delimiter=' ')

# Inspecting test data
#test_data[0:0]

# Approach:
# 1. For each document in test data we have to predict it's class
# 2. Algorithm: Add up probabilities corresponding to those words which exisit in the document. This gives log(PROD(p(x|y)). This has to be multipled by p(y) for that document class.


test_docids = pd.unique(test_data.docid.ravel())

labels = np.array(range(1,21))

# Creating a matrix that contains docids to labels mapping
# Cells will be filled with log(p(y|x)) values ahead
label_pred_matrix = pd.DataFrame(0, index=test_docids, columns=labels)
#label_pred_matrix    # docids are rows, labels are columns and we want to fill in this matrix

print "Starting to fill log(p(y|x)) or label_pred_matrix"
# This portion fills up the label prediction matrix with log(PROD(p(x|y)))
for row_id in test_data.index:
    docid, wid, count = test_data.loc[row_id]

    # Just to check how fast are we going. And how much time remaning
    if docid%100 == 0:
        print docid
    
    label_pred_matrix.loc[docid] += matrix.loc[wid]
    # For each label assumption calculate p(x|y) and fill in prediction matrix
    #for label_assumption in range(1,21):
        #label_pred_matrix[label_assumption][docid] += matrix[label_assumption][wid]

label_pred_matrix


# Adding log priors to pred matrix i.e. log(p(y)) which will
# make the net sum = log(p(y|x))
for l in range(1,21):
    label_pred_matrix[l] += label_priors['log_priors'][l]


predictions = label_pred_matrix.apply(np.argmax, axis=1)

# Saving the predictions (docid to predicted label mapping as csv)
# which will be used to compute accuracy and confusion matrix
predictions.to_csv('predictions_naive_bayes_multinomial_alphas.txt')
