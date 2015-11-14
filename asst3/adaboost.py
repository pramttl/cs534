
# coding: utf-8

from __future__ import division
import math
import numpy as np
import pandas as pd
from numpy import genfromtxt
from pprint import pprint

train_data = genfromtxt('SPECT-train.csv', delimiter=',')
test_data = genfromtxt('SPECT-test.csv', delimiter=',')

# Transforming dataset replacing 0 by -1
# - Makes it convenient to get final hypotheisis just by applying sign function
# - Class slides explaining adaboost assumed +1, -1 lablels
for row in train_data:
    row[row==0] = -1

for row in test_data:
    row[row==0] = -1

# Label possibilities
POS = 1
NEG = -1

nexamples = train_data.shape[0]
nfeatures = train_data.shape[1] - 1    # First col is class label

traindf = pd.DataFrame(train_data)
testdf = pd.DataFrame(test_data)
traindf


# Function for initializing distribution weights
def initialize_D(train_data):
    return np.tile(float(1)/train_data.shape[0], train_data.shape[0])
                           

def get_entropy(p, n, counts=True):
    """
    :p: Counts of postitives or net weight of positive examples
    :n: Counts or net weight of negative examples
    
    counts=False implies pf and nf are fractions of postive and negative examples respectively
    instead of counts (They can also be weight of postitives and negatives)
    
    :return: Entropy float value
    Assuming: 
    - Only postive or negative examples
    - If counts==False, then sum(weight_positives) + sum(weight_negatives) must be 1
      To be on the safer side it's better to let counts == True with weights
    """
    if (p == 0) or (n == 0):
        return float(0)
    else:
        if counts:
            p, n = float(p)/(p+n) , float(n)/(p+n)
        return -p*math.log(p,2) - n*math.log(n,2)

    
# Testing the get_entropy function for some key cases so that I am sure they work :)
assert get_entropy(1/2,1/2) == float(1)
assert get_entropy(1,0) == float(0)
assert get_entropy(0,1) == float(0)

assert get_entropy(2,2,True) == float(1)
assert get_entropy(4,0,True) == float(0)
assert get_entropy(0,4,True) == float(0)


def best_feature(train_data, D):
    """
    Returns tuple:
    split_feature, left_pos_weight, left_neg_weight, right_pos_weight, right_neg_weight, LE, RE
    
    1. split_feature: Which feature to create split at for max information gain
    
    (Following weights are returned corresponding to best feature)
    2. left_pos_weight   # sum(weight of positive examples in left branch)
    3. left_neg_weight
    4. right_pos_weight
    5. right_neg_weight
    6. LE    # Left Entropy
    7. RE    # Right Entropy

    These quantities are helpful in understanding what prediction to make.
    """
    least_split_entropy = None
    nexamples = train_data.shape[0]
    nfeatures = train_data.shape[1] - 1

    # Iterating over each feature and seeing if it corresponds to best decision stump
    for f in range(1,nfeatures+1):

        # For left branch there will be pos and neg examples. Each example has a weight
        left_pos_weight = 0    # Will store sum(weight of positive examples in left branch)
        left_neg_weight = 0
        right_pos_weight = 0
        right_neg_weight = 0

        # Iterating over each example
        for e in range(nexamples):

            # Making sure each row has correct num of columns
            assert train_data[e].shape[0] == nfeatures + 1

            # Left branch of stump at feature f
            if train_data[e][f] == NEG:
                # If NEG example then neg weight increases
                if train_data[e][0] == NEG:
                    left_neg_weight += D[e]
                    # left_neg_weight += 1  # This would be the case in simple decision stump learner without weights

                # If POS example then pos weight increases
                else:
                    left_pos_weight += D[e]

            # Right branch of stump at feature f
            else:
                if train_data[e][0] == NEG:
                    right_neg_weight += D[e]
                else:
                    right_pos_weight += D[e]

        # Left and right branch entropies. Treat left weights as counts as total weight does
        # not add upto one for individual branch
        LE = get_entropy(left_pos_weight, left_neg_weight)     # Left Entropy
        RE = get_entropy(right_pos_weight, right_neg_weight)   # Right Entropy 
        LW = left_pos_weight + left_neg_weight                 # Left Weight
        RW = right_pos_weight + right_neg_weight               # Right Weight

        split_entropy = LW*LE + RW*RE

        #print split_entropy
        if not least_split_entropy:
            least_split_entropy = split_entropy
            best_feature = f
            LR_entropies = LE, RE   # To store best left and right entropies

        # If some feature gives a better split entropy then that feature becomes current best feature
        if split_entropy < least_split_entropy:
            least_split_entropy = split_entropy
            best_feature = f
            LR_entropies = LE, RE

    return best_feature, left_pos_weight, left_neg_weight, right_pos_weight, right_neg_weight, LE, RE

############### Checking if a simple decision stump answer is consistent with part 1 #################
# Find best decision stump
# Since distribution is uniform initially this is same as part 1, i.e. finding best decision stump
print "Checking best feature to split at for first decision stump, this should be 13 as we observed in part 1 of question"
D = initialize_D(train_data)
print best_feature(train_data, D)
######################################################################################################


def mark_error_examples(actual_labels, predicted_labels):
    """
    Marks error examples as 1 and rest as 0 and returns a new array
    The new array has 1 at the indices where there was an error
    and 0 at indices where there was not.
    """
    nexamples = len(actual_labels)
    assert nexamples == len(predicted_labels)
    error_array = np.zeros(nexamples)
    for i in range(nexamples):
        if actual_labels[i] != predicted_labels[i]:
            error_array[i] = 1
    return error_array


def learn(train_data, D):
    """
    :return: hypothesis_array, weighted_error, accuracy 
    
    hypothesis_array: An array that contains predicted value for each input value in given dataset.
    Array size: nexamples (Each value is either 0 or 1)
    
    error_array: Boolean array. Same size as hypothesis array except that this array contains True at
    those indices where there was a mismatch from actual label
    
    weighted_error: Epsilon in adaboost algorithm, used to calculate alpha and new distribution
    
    accuracy: Accurracy of decision stump w.r.t training data itself; i.e. Trg accuracy
    """
    input_data = train_data
    
    split_feature, left_pos_weight, left_neg_weight, right_pos_weight, right_neg_weight, LE, RE = best_feature(train_data, D)

    hypothesis_array = np.zeros(train_data.shape[0])
    nelements = input_data.shape[0]
    
    # Now that we know which faeture to split at, next step is to understand whether each branch predicts positive
    # or negative; we should note that both branches may have the same prediction

    if left_pos_weight > left_neg_weight:
        left = POS
    else:
        left = NEG
    
    if right_pos_weight > right_neg_weight:
        right = POS
    else:
        right = NEG

    for e in range(nelements):
        if input_data[e][split_feature] == NEG:
        # Left branch
            hypothesis_array[e] = left
        else:
        # Right branch
            hypothesis_array[e] = right

    error_array = mark_error_examples(input_data[:,0], hypothesis_array)
    # Contains 1 at mismatches or misclassifified examples, rest examples are marked 0
    
    # [IMP] Weighted_error (Epsilon used in Adaboost)
    weighted_error = sum([D[i] for i in error_array if i==1]) / sum(D) # Sum of weights of misclassified / Net weight=1

    error_fraction = sum(error_array)/nelements
    accuracy = 1 - error_fraction
    return hypothesis_array, error_array, weighted_error, accuracy

D = initialize_D(train_data)
hypothesis_array, error_array, epsilon, accuracy = learn(train_data, D)
hypothesis_array, error_array, epsilon, accuracy


def update_D(D, epsilon, error_array):
    """
    Returns D_i+1 from D_i. Normalization is included in this funciton.
    """
    D_new = np.zeros(D.shape)
    for e,val in enumerate(error_array):
        
        if val == 1:
            # D update for mismatch
            D_new[e] = D[e] * math.sqrt((float(1)/epsilon) - 1)
        else:
            # D update for correct predictions
            D_new[e] = D[e] / math.sqrt((float(1)/epsilon) - 1)

    # Normalizing so that distribution weights add upto 1
    Z = sum(D_new)
    D_new = D_new/Z
    return D_new

# For storing alpha at each ensemble iteration
# We need alphas in end for final hypothesis
alphas = []
hypothesis_arrays = []

print "==============================="
print "Starting Adaboost now...."
print "==============================="

D = initialize_D(train_data)
error_indices = None
print "Initial D"
print D
L = 10 # Ensemble size
for l in range(L):
    #print "--------", l, "--------"
    
    # This prints the sum of misclassified examples of previous step
    #if error_indices:
    #    print sum([D[i] for i in error_indices])
    
    hypothesis_array, error_array, epsilon, accuracy = learn(train_data, D)
    alpha = 0.5 * math.log((1/epsilon) - 1)
    alphas.append(alpha)
    hypothesis_arrays.append(hypothesis_array)
    D = update_D(D, epsilon, error_array)

    #error_indices = [i for i,e in enumerate(error_array) if e==1]

    print "alpha, epsilon, sum(D)"
    print alpha, epsilon, sum(D)
    print "D"
    print D
    #pprint([(i, D[i]) for i in error_indices])


hypothesis_arrays

# ----------------------------------------------------------------
# Adaboost training done, alphas and individual hypotheis obtained
# Now we have to use alphas and H to make final hypothesis

s = np.zeros(nexamples)
for i in range(L):
    s += alphas[i] * hypothesis_arrays[i]
s

H = np.sign(s)

print "----------------------------------------------------------"
print "Final hypothesis for all %d examples"%(train_data.shape[0])
print "----------------------------------------------------------"
print H