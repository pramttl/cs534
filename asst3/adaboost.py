
# coding: utf-8

from __future__ import division
import math
import numpy as np
from numpy import genfromtxt
from pprint import pprint

#X = genfromtxt('toy_train.csv', delimiter=',')
train_data = genfromtxt('SPECT-train.csv', delimiter=',')
test_data = genfromtxt('SPECT-test.csv', delimiter=',')

# Label possibilities
POS = 1
NEG = -1

# Transforming dataset replacing 0 by -1
# - Makes it convenient to get final hypotheisis just by applying sign function
# - Class slides explaining adaboost assumed +1, -1 lablels
for row in train_data:
    row[row==0] = NEG

for row in test_data:
    row[row==0] = NEG


nexamples = train_data.shape[0]
nfeatures = train_data.shape[1] - 1    # First col is class label


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
        return -p*math.log(p,2) -n*math.log(n,2)


def get_root_entropy(X, D, counts=True):
    pos_weight = 0
    neg_weight = 0
    for e in range(X.shape[0]):
        if X[e][0] == POS:
            pos_weight += D[e]
        else:
            neg_weight += D[e]
    root_entropy = get_entropy(pos_weight, neg_weight)
    #print "pos_weight, neg_weight, root_entropy"
    #print pos_weight, neg_weight, root_entropy
    return root_entropy


def get_information_gain(split_entropy, X, D):
    root_entropy = get_root_entropy(X, D)
    gain = root_entropy - split_entropy
    return gain


# Function for initializing distribution weights
def initialize_D(train_data):
    return np.tile(float(1)/train_data.shape[0], train_data.shape[0])


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


class DecisionStump():

    split_feature = None
    left = None
    right = None
    #hypothesis_array = None

    def fit(self, X, y, D):
        max_split_gain = float('-inf')   # Setting to lowest possible value
        nexamples = X.shape[0]
        nfeatures = X.shape[1]

        # Iterating over each feature and seeing if it corresponds to best decision stump
        for f in range(0, nfeatures):

            # For left branch there will be pos and neg examples. Each example has a weight
            left_pos_weight = 0    # Will store sum(weight of positive examples in left branch)
            left_neg_weight = 0
            right_pos_weight = 0
            right_neg_weight = 0

            # Iterating over each example
            for e in range(nexamples):

                # Left branch of stump at feature f
                if X[e][f] == NEG:
                    # If NEG example then neg weight increases
                    if y[e] == NEG:
                        left_neg_weight += D[e]
                        # left_neg_weight += 1  # This would be the case in simple decision stump learner without weights

                    # If POS example then pos weight increases
                    else:
                        left_pos_weight += D[e]

                # Right branch of stump at feature f
                else:
                    if y[e] == NEG:
                        right_neg_weight += D[e]
                    else:
                        right_pos_weight += D[e]


            # Left and right branch entropies. Treat left weights as counts as total weight does
            # not add upto one for individual branch
            LE = get_entropy(left_pos_weight, left_neg_weight)     # Left Entropy
            RE = get_entropy(right_pos_weight, right_neg_weight)   # Right Entropy 
            LW = left_pos_weight + left_neg_weight                 # Left Weight
            RW = right_pos_weight + right_neg_weight               # Right Weight

            #split_entropy = (LW/(LW+RW))*LE + (RW/(LW+RW))*RE    # LW + RW = 1.0
            split_entropy = (LW/(LW+RW))*LE + (RW/(LW+RW))*RE    # LW + RW = 1.0
            curr_split_gain = get_information_gain(split_entropy, X, D)

            #print "(1) f, e, left_pos_weight, left_neg_weight, right_pos_weight, right_neg_weight, LE, RE, split_entropy"
            #print f, e, left_pos_weight, left_neg_weight, right_pos_weight, right_neg_weight, LE, RE, split_entropy
            #print

            # If some feature gives a better split entropy then that feature becomes current best feature
            if curr_split_gain > max_split_gain:
                max_split_gain = curr_split_gain
                best_feature = f
                best_left_pos_weight = left_pos_weight
                best_left_neg_weight = left_neg_weight
                best_right_pos_weight = right_pos_weight
                best_right_neg_weight = right_neg_weight
                LR_entropies = LE, RE

        self.split_feature = best_feature

        # Now that we know best feature, let's find best label
        ############################ Part 2 #######################################
        # 4 cases, (0,1), (1,0), (0,0), (1,1)
        cases = [(NEG,POS), (POS,NEG), (NEG,NEG), (POS,POS)]
        hypothesis_array_cases = [np.zeros(nexamples), np.zeros(nexamples), np.zeros(nexamples), np.zeros(nexamples)]
        
        weighted_error = float('inf')
        for i, case in enumerate(cases):
            for e in range(nexamples):
                if X[e][self.split_feature] == NEG:
                # Left branch
                        hypothesis_array_cases[i][e] = case[0]   # case[0] = left
                else:
                # Right branch
                    hypothesis_array_cases[i][e] = case[1]

            assert len(hypothesis_array_cases[i]) == nexamples
            error_array = mark_error_examples(y, hypothesis_array_cases[i])
            # Contains 1 at mismatches or misclassifified examples, rest examples are marked 0
        
            # [IMP] Weighted_error (Epsilon used in Adaboost)
            curr_weighted_error = sum([D[j] for j,error in enumerate(error_array) if error==1]) # Sum of weights of misclassified / Net weight=1
            
            #print "error_array"
            #print error_array
            #print "casei, curr_weighted_error"
            #print i, curr_weighted_error
            if curr_weighted_error <= weighted_error:
                weighted_error = curr_weighted_error
                best_error_array = error_array
                best_case_index = i
                best_case = case

        self.left = best_case[0]
        self.right = best_case[1]
        #self.hypothesis_array = hypothesis_array_cases[best_case_index]
        return True


    def predict(self, X):
        pred = []
        for row in X:
            if row[self.split_feature] == NEG:
                pred.append(self.left)
            else:
                pred.append(self.right)
        return np.array(pred)


def update_D_epsi(D, epsilon, error_array):
    """
    Returns D_i+1 from D_i. Normalization is included in this funciton.
    """
    D_new = np.zeros(D.shape)
    for e,val in enumerate(error_array):
        
        if val == 1:
            # D update for mismatch
            D_new[e] = D[e] * math.sqrt(1/epsilon - 1)
        else:
            # D update for correct predictions
            D_new[e] = D[e] / math.sqrt(1/epsilon - 1)

    # Normalizing so that distribution weights add upto 1
    Z = sum(D_new)
    D_new = D_new/Z
    return D_new


def update_D(D, alpha, error_array):
    """
    Returns D_i+1 from D_i. Normalization is included in this funciton.
    """
    D_new = np.zeros(D.shape)
    for e,val in enumerate(error_array):
        
        if val == 1:
            # D update for mismatch
            D_new[e] = D[e] * math.exp(alpha)
        else:
            # D update for correct predictions
            D_new[e] = D[e] * math.exp(-alpha)

    # Normalizing so that distribution weights add upto 1
    Z = sum(D_new)
    D_new = D_new/Z
    return D_new


print "==============================="
print "====== Starting Adaboost ======"
print

ENSEMBLE_SIZE_CASES = range(1, 30)
for L in ENSEMBLE_SIZE_CASES:

    # For storing alpha at each ensemble iteration
    # We need alphas in end for final hypothesis
    alphas = []
    hypothesis_arrays = []
    features = []
    decision_stumps = []

    D = initialize_D(train_data)

    for l in range(L):

        h1 = DecisionStump()
        X = train_data[:,1:]    # Train data excl. first ground truth col
        y = train_data[:,0]     # Actual labels
        h1.fit(X, y, D)

        # Append after fitting so that it can be used when calculating test error
        decision_stumps.append(h1)

        hypothesis_array = h1.predict(X)
        error_array = mark_error_examples(y, hypothesis_array)
        hypothesis_array != y
        epsilon = D.dot(error_array)

        alpha =  0.5 * math.log((1-epsilon)/epsilon)
        alphas.append(alpha)
        hypothesis_arrays.append(hypothesis_array)
        #D = update_D_epsi(D, epsilon, error_array)
        D = update_D(D, alpha, error_array)

        #print sum(error_array), sum(hypothesis_array)
        #print "%d:, %d"%(l, epsilon)
        #print
        #print "D"
        #print D
        #print "error_array"
        #print error_array
        #print "hypothesis_array"
        #print hypothesis_array


    # ----------------------------------------------------------------
    # Adaboost training done, alphas and individual hypotheis obtained
    # Now we have to use alphas and H to make final hypothesis

    s = np.zeros(train_data.shape[0])
    for i in range(L):
        s += alphas[i] * hypothesis_arrays[i]

    H = np.sign(s)

    #print "----------------------------------------------------------"
    #print "Final hypothesis for all %d examples"%(X.shape[0])
    #print "----------------------------------------------------------"
    #print H
    a = train_data[:,0]
    error_array = mark_error_examples(a, H)
    #print "error_array"
    #print error_array
    train_accuracy = (1 - sum(error_array)/train_data.shape[0])


    s2 = 0
    for l in range(L):
        stump = decision_stumps[l]
        ha = stump.predict(test_data[:,1:])
        s2 += alphas[l] * ha


    #print "----------------- Test Hypothesis & Accuracy -------------------"
    H2 = np.sign(s2)
    #print H2
    a2 = test_data[:,0]
    error_array2 = mark_error_examples(a2, H2)
    test_accuracy = (1 - sum(error_array2)/test_data.shape[0])

    if L==1:
        print "ensemble_size\ttrain_error\ttest_error"
    print "%d\t%f\t%f\t"%(L, 1-train_accuracy, 1-test_accuracy)
