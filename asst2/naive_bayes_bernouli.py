
# coding: utf-8

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
train_data['doclabel'] = train_data_labels


grouped = train_data.groupby(['doclabel', 'wid'])
#grouped.groups
s = grouped.apply(len)
t = s.unstack()
t.shape
del s


t = pd.concat([t, pd.DataFrame(columns=range(t.shape[1]+1, VOCAB_SIZE+1))])
t.fillna(0, inplace=True)


matrix = t.transpose()
matrix.shape

# Currently matrix just contains counts
matrix


# Laplace smoothing
for l in range(1, len(labels)+1):
    # Laplace smoothing, column by column (or in other words label by label)
    matrix[l] = matrix[l].apply(lambda x: math.log((x+1) / (doccount_from_label(l) + 2)))

# Matrix with log probability

#################### Ready to make predictions #############

# Loading test data
test_data = pd.read_csv('test.data', delimiter=' ')
test_doc_labels = pd.read_csv('test.label', delimiter=' ')

#test_data[0:5]


# 1. For each document in test data we have to predict it's class
# 2. Algorithm: Add up probabilities corresponding to those words which exisit in the document. This gives log(PROD(p(x|y)). This has to be multipled by p(y) for that document class.


# Get all unique document id's into an array
test_docids = pd.unique(test_data.docid.ravel())


# label_pred_matrix
labels = np.array(range(1,21))
label_pred_matrix = pd.DataFrame(0, index = test_docids, columns = labels)
label_pred_matrix    # docids are rows, labels are columns and we want to fill in this matrix


# Convert data frame to array (might be more optimized than iterating over rows in data frame)
tdata = np.array(test_data)
tdata


############ Warning!! This might take a while #########
for row in tdata:
    docid, wid, count = row

    # Printing out status for every 10'th document being processed
    if docid%10 == 0:
        print docid
    
    # For each label assumption calculate p(x|y) and fill in prediction matrix
    for label_assumption in range(1,21):
        label_pred_matrix[label_assumption][docid] += matrix[label_assumption][wid]

# Now this matrix is only filled with log(PROD(p(x|y))), need to add log(p(y))
label_pred_matrix


# Adding prior log probabilities
for l in range(1,21):
    label_pred_matrix[l] += label_priors['log_priors'][l]


# Creating prediction series
predictions = label_pred_matrix.apply(np.argmax, axis=1)

