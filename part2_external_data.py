import numpy as np
import os
import matplotlib.pyplot as plt
import cv2

DataDir = "D:/Tensorflow_Sentdex/data/"
Categories = ["Dog", "Cat"]

IM_SIZE = 30

training_data = []

def create_train():
    for category in Categories:
        path = os.path.join(DataDir, category)
        classnum = Categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IM_SIZE, IM_SIZE))
                training_data.append([new_array,classnum])
            except Exception as e:
                pass

create_train()
print("hello")
print(len(training_data))

import random
random.shuffle(training_data)
for sample in training_data[:10]:
    print(sample[1])

X = []
Y = []

for features ,label in training_data:
    X.append(features)
    Y.append(label)
X = np.array(X).reshape(-1, IM_SIZE,IM_SIZE,1)

import pickle

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("Y.pickle","wb")
pickle.dump(Y, pickle_out)
pickle_out.close()

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("Y.pickle","rb")
Y = pickle.load(pickle_in)
print(X[1])
