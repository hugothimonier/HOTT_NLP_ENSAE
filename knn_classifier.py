import numpy as np
import progressbar

def predict(neighbor_classes, C):
    # Make sure all classes are considered
    labels = np.concatenate((neighbor_classes, np.arange(C)-1))
    # Find class frequency among neighbors
    weights = np.unique(labels, return_counts=True)[1]
    # Find most popular class
    prediction = np.argmax(weights)

    # If most popular class is ambiguous try with fewer neighbors; else return
    if sum(weights[prediction] == weights) > 1:
        return predict(neighbor_classes[:-1], C)
        
    else:
        return np.unique(labels)[prediction]



def knn(X_train, X_test, y_train, y_test, method, C, n_neighbors=7):
    # Number of classes
    n_classes = len(np.unique(y_train))

    prediction = []
    for doc in X_test:
        doc_to_train = [method(doc, x, C) for x in X_train]
        # Find indices of n_neighbors closest documents
        rank = np.argsort(doc_to_train)[:n_neighbors]

        # Make prediction based on most popular class among neighbors
        prediction.append(predict(y_train[rank], n_classes))

    # Print and return test error
    test_error = 1 - (prediction == y_test).mean()
    return test_error, prediction

#If we want to do hyperparameter tuning calculazte once and for one the distance matrix
def calculate_matrix_dist(X_train, X_test ,method ,C):

    mat=[]
    for i, doc in enumerate(progressbar.progressbar(X_test)):
        mat.append([method(doc, x, C) for x in X_train])
    return mat

def knn2(mat, y_train, y_test, n_neighbors=7):
    # Number of classes
    n_classes = len(np.unique(y_train))

    prediction = []
    for i, cat in enumerate(y_test):
        doc_to_train = mat[i]
        # Find indices of n_neighbors closest documents
        rank = np.argsort(doc_to_train)[:n_neighbors]

        # Make prediction based on most popular class among neighbors
        prediction.append(predict(y_train[rank], n_classes))

    # Print and return test error
    test_error = 1 - (prediction == y_test).mean()
    return test_error, prediction