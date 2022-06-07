import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np

def getDataset():
    print(tf.__version__)
    print(tf.config.list_physical_devices('GPU'))
    print(tf.test.gpu_device_name())

    final_data = pd.read_csv("./dataset.csv")
    final_data = final_data.drop(labels='filename', axis=1)
    fit = StandardScaler()
    class_list = final_data.iloc[:, -1]
    convertor = LabelEncoder()
    y = convertor.fit_transform(class_list)
    X = fit.fit_transform(np.array(final_data.iloc[:, :-1], dtype=float))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    return X_train, X_test, y_train, y_test

def plotValidate(history):
    pd.DataFrame(history.history).plot(figsize=(12 ,6))
    plt.show()