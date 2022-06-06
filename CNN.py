# reference: https://www.kaggle.com/code/aishwarya2210/let-s-tune-the-music-with-cnn-xgboost

import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras

def plotValidate(history):
    print("Validation Accuracy" ,max(history.history["val_accuracy"]))
    pd.DataFrame(history.history).plot(figsize=(12 ,6))
    plt.show()

def trainModel(X_train, X_test,y_train, y_test, model, epochs, optimizer):
    batch_size = 128
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                   metrics='accuracy'
    )
    return model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs,
                     batch_size=batch_size)

def getModel(X_train):
    model = keras.models.Sequential([
        keras.layers.Dense(512, activation="relu", input_shape=(X_train.shape[1],)),
        keras.layers.Dropout(0.2),

        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dropout(0.2),

        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.2),

        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.2),

        keras.layers.Dense(10, activation="softmax"),

    ])
    return model