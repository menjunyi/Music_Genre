# reference

from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np
from SVM import SVM
from sklearn import preprocessing
import numpy as np
import pickle
import sklearn.metrics as metrics

def safeLoad(filename):
    return pickle.load(open(filename, 'rb'))

def safeKeep(obj, filename):
    with open(filename, 'wb') as op:
        pickle.dump(obj.op, pickle.HIGHEST_PROTOCOL)

def test():
    pass

if __name__ == "__main__":
    X, y = train_test_split()

    svm = SVM(n_iters=100)
    svm.fit(X,y)
    result = svm.predit(X)
    X = np.array(safeLoad('trainvec.pkl'))
    Y = safeLoad('labels.pkl')
    testX = np.array(safeLoad('testvec.pkl'))
    testY = np.array(safeLoad('testlabel.pkl'))

    clf = SVM()

    X = preprocessing.scale(X)
    testX = preprocessing.scale(testX)

    clf.fit(X, Y)

    predVal = clf.predict(testX)

    pop = []
    jazz = []
    metal = []
    classical = []

    for i, val in enumerate(testY):
        if val == 'pop':
            pop.append([val, predVal[i]])
        elif val == 'jazz':
            jazz.append([val, predVal[i]])
        elif val == 'metal':
            metal.append([val, predVal[i]])
        else:
            classical.append([val, predVal[i]])

    pop_accuracy = metrics.accuracy_score(pop[:][0], pop[:][1])
    jazz_accuracy = metrics.accuracy_score(jazz[:][0], jazz[:][1])
    metal_accuracy = metrics.accuracy_score(metal[:][0], metal[:][1])
    classical_accuracy = metrics.accuracy_score(classical[:][0], classical[:][1])

    print("Pop accuracy:", pop_accuracy)
    print("Jazz accuracy:", jazz_accuracy)
    print("Metal accuracy:", metal_accuracy)
    print("Classical accuracy:", classical_accuracy)
    print("Overall accuracy:", metrics.accuracy_score(testY, predVal))
