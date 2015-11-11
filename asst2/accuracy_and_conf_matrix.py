# -*- coding: utf-8 -*-

"""
Reads in predictions file which contains docid, prediction_label in each row.
Also reads in actual labels from test.label file.
Uses the above two to generate accuracy and confusion matrix and prints to stdout.
"""

from __future__ import division 
import numpy as np
from numpy import genfromtxt

test_label = genfromtxt('test.label', delimiter='\n') 
test_prediction = genfromtxt('data/predictions_naive_bayes_multinomial_3.txt', delimiter=',') 

#total number of documents
total_num_test=len(test_label)
#number of correctly classified documents 
#true positive
tp = 0
for row in range(len(test_label)):
    if test_label[row] == test_prediction[row,1]:
        tp +=1
    
# overall accuracy    
accuracy = tp/total_num_test
        
# confusion table
ct = np.zeros(shape=(21,21))
for row in range(len(test_label)):
    if test_label[row] == test_prediction[row,1]:
        ct[test_label[row],test_label[row]] += 1
    else:
        ct[test_label[row],test_prediction[row,1]] += 1

# Printing confusion matrix
for i, r in enumerate(ct):
    if i==0:
        continue
    for j, c in enumerate(r):
        if j==0:
            continue
        print str(c) + '\t',
    print

# Printing accuracy
print accuracy