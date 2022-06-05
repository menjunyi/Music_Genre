import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
# reference: https://www.kaggle.com/code/aishwarya2210/let-s-tune-the-music-with-cnn-xgboost

print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))
print(tf.test.gpu_device_name())


final_data = pd.read_csv("./dataset.csv")
final_data = final_data.drop(labels='filename',axis=1)
fit = StandardScaler()
class_list = final_data.iloc[:, -1]
convertor = LabelEncoder()
y = convertor.fit_transform(class_list)
X = fit.fit_transform(np.array(final_data.iloc[:, :-1], dtype = float))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

def plotValidate(history):
    print("Validation Accuracy" ,max(history.history["val_accuracy"]))
    pd.DataFrame(history.history).plot(figsize=(12 ,6))
    plt.show()

def trainModel(model, epochs, optimizer):
    batch_size = 128
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                   metrics='accuracy'
    )
    return model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs,
                     batch_size=batch_size)

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
print(model.summary())
model_history = trainModel(model=model, epochs=1500, optimizer='adam')


test_loss, test_accuracy = model.evaluate(X_test, y_test, batch_size=128)
print("The test loss is :",test_loss)
print("\nThe test Accuracy is :",test_accuracy*100)

#Plot the loss & accuracy curves for training & validation
plotValidate(model_history)