# reference
from sklearn.svm import NuSVC
from SVM import SVM
import CNN
import utils

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = utils.getDataset()
    # my SVM test
    svm = SVM(n_iters=100)
    svm.fit(X_train,y_train)
    predict_my = svm.predict(X_test)
    print("my prediction accuracy:", (predict_my == y_test).sum() / predict_my.shape[0])
    # utils.plotValidate(model_history)

    # sklearn svm test
    clf = NuSVC(kernel='rbf', nu = 0.00001, degree = 5)
    clf.fit(X_train,y_train)
    predict_nusvc = clf.predict(X_test)
    print("sklearn prediction accuracy:", (predict_nusvc == y_test).sum() / predict_nusvc.shape[0])

    # CNN test
    model = CNN.getModel(X_train)
    print(model.summary())
    model_history = CNN.trainModel(X_train, X_test, y_train, y_test, model=model, epochs=500, optimizer='adam')

    test_loss, test_accuracy = model.evaluate(X_test, y_test, batch_size=128)
    print("The test loss is :",test_loss)
    print("\nThe test Accuracy is :",test_accuracy*100)
    print("Validation Accuracy" ,max(model_history.history["val_accuracy"]))

    #Plot the loss & accuracy curves for training & validation
    utils.plotValidate(model_history)
