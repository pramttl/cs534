import numpy as np
from numpy import genfromtxt

def grad(w, X, y, lam):
    """
    The gradient of the linear regression with l2 regularization cost function

    :param w: Weights vector  Shape: (d+1,1)
    :param X: Input dataset. N X d+1 matrix where N is number of examples, d is number of features
    :param y: The vector corresponding to ground truth or output vector corresponding to N examples. (N X 1)
    :param lam: Regularization factor
    :return: The gradient of the linear regression with l2 regularization cost function
    """
    ans = 0

    N = X.shape[0]
    assert X.shape[0] == y.shape[0]

    for i in range(0, N):
        ans += 2 * (np.dot(w, X[i]) - y[i]) * X[i]

    assert ans.shape[0] == w.shape[0]

    ans += 2 * lam * w

    return ans


def make_predictions(w, dataset):
    """
    Takes in weights vector w, and a dataset matrix which contains input rows excluding output column
    and returns prediction vector.
    """
    pred = np.array([])
    for row in dataset:
        # test_without_gt_ = test_with_gt_col[:45]    # Ground truth column removing while creating test vector
        pred_value = np.dot(row, w)
        pred = np.append(pred, pred_value)
    return pred


def sse(gt, pred):
    """
    Get SSE given ground truth and prediction vectors
    """
    return sum((gt-pred)**2)


def norm_train_data(train_data2):
    """
    Normalize training data
    :train_data2: Is the training data without the last output column
    """
    NORMALIZED_TRAIN_DATA = np.zeros(train_data2.shape)
    mean_array = np.zeros(train_data2.shape[1])
    min_array = np.zeros(train_data2.shape[1])
    max_array = np.zeros(train_data2.shape[1])

    # Normalizing each column in training dataset input
    for c in range(train_data2.shape[1]):
        original_vector = train_data2[:,c]

        mean_array[c] = np.mean(original_vector)
        max_array[c] = np.max(original_vector)
        min_array[c] = np.min(original_vector)

        if max_array[c] == min_array[c]:
            # Don't normalize as it is most likely first col only
            NORMALIZED_TRAIN_DATA[:,c] = np.ones((NORMALIZED_TRAIN_DATA[:,c]).shape)
        else:
            new_vector = (original_vector - mean_array[c]) / (max_array[c] - min_array[c])
            NORMALIZED_TRAIN_DATA[:,c] = new_vector.transpose()

    return NORMALIZED_TRAIN_DATA, min_array, mean_array, max_array


def norm_test_data(test_data2, min_array, mean_array, max_array):
    """
    This is a separate function than normalized train data because it does not find mean, min max on test data
    but uses same min, mean, max of training data for respective columns.
    """
    NORMALIZED_TEST_DATA = np.zeros(test_data2.shape)     # Will eventually store normalized test data

    # Normalize the test data set (albeit using training data mean, min max for respective cols)
    for c in range(test_data2.shape[1]):
        col_vector = test_data2[:,c]

        if max_array[c] == min_array[c]:
            # Don't normalize as it is most likely first col only
            NORMALIZED_TEST_DATA[:,c] = np.ones((NORMALIZED_TEST_DATA[:,c]).shape)
        else:
            new_col_vector = (col_vector - mean_array[c]) / (max_array[c] - min_array[c])
            NORMALIZED_TEST_DATA[:,c] = new_col_vector.transpose()

    return NORMALIZED_TEST_DATA


def grad_descent(NORMALIZED_TRAIN_DATA, y, alpha, lam, convg_criteria=10**(-3)):
    """
    Performs the gradient descent. Runs 10^7 iterations max and if no covenrgence
    till that time returns None
    """
    w = np.zeros(45)
    # Running gradient descent algorithm
    for runs in xrange(10000000):
        g = grad(w, NORMALIZED_TRAIN_DATA, y, lam)
        w = w - (alpha * g)
        normg = np.linalg.norm(g)   # Maginitude of gradient
        print "g: ", np.linalg.norm(g)

        # Convergence criteria
        if normg <= convg_criteria:
            break

        # Rough cnvergence check to avoid redundant computation
        if runs >= 10**(4) and normg > 10:
            # Even after so many iterations there is no convergence
            # So we should stop doing grad descent here as convergence is not likely
            # in a reasonable amount of time
            return normg, w
    return normg, w

def crossval_data(d):
    """
    d: Training dataset
    Returs 10 cases each having training dataset and test dataset using 10 way cross validation technique
    """
    eyes = range(0, 100, 10)
    cases = []
    for i in eyes:
        trgdata = np.vstack((d[0:i,:], d[i+10:,:]))
        testdata = d[i:i+10,:]
        cases.append({"trgdata": trgdata, "testdata": testdata})
    return cases

alpha = 0.009
lam = 0

train_data = genfromtxt('train.csv', delimiter=',')
cases = crossval_data(train_data)

for case in cases:
    NORMALIZED_TRAIN_DATA, min_array, mean_array, max_array = norm_train_data(case["trgdata"][:,:45])
    y = case["trgdata"][:, 45]
    normg, w = grad_descent(NORMALIZED_TRAIN_DATA, y, alpha, lam, convg_criteria=10**(-3))

    if normg > 10**(-3):
        print "No convergence"
        case["test_pred"] = None
        case["test_sse"] = None
        case["w"] = None
        continue

    NORMALIZED_TEST_DATA = norm_test_data(case["testdata"][:,:45], min_array, mean_array, max_array)

    test_pred = make_predictions(w, NORMALIZED_TEST_DATA)
    test_sse = sse(case["testdata"][:,45], test_pred)
    case["test_pred"] = test_pred
    case["test_sse"] = test_sse
    case["w"] = w
    case["normg"] = normg
    print test_sse


s = 0
for case in cases:
    s += case['test_sse']
print s

"""
test_data = genfromtxt('test.csv', delimiter=',')

train_data2 = train_data[:, 0:45]
NORMALIZED_TRAIN_DATA, min_array, mean_array, max_array = norm_train_data(train_data2)

# NORMALIZED_TRAIN_DATA is normalized data

# Running gradient descent algorithm
normg, w = grad_descent(NORMALIZED_TRAIN_DATA, y, alpha, lam, convg_criteria=10**(-3))

print "normg: ", normg
print "weights: ", w

############################

test_data2 = test_data[:,:45]     # Test data exc. ground truth column
NORMALIZED_TEST_DATA = norm_test_data(test_data2, min_array, mean_array, max_array)

print "NORMALIZED_TRAIN_DATA", NORMALIZED_TRAIN_DATA
print "NORMALIZED_TEST_DATA", NORMALIZED_TEST_DATA

############################

test_pred = make_predictions(w, NORMALIZED_TEST_DATA)
test_sse = sse(test_data[:,45], test_pred)
print "test_sse:  ",  test_sse

# Predictions on training data
train_pred = make_predictions(w, NORMALIZED_TRAIN_DATA)
train_sse = sse(train_data[:,45], train_pred)

print "train_sse: ", train_sse
# cross-validation

"""
