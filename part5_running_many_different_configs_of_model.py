
import tensorflow as tf
from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout,Activation,Flatten,Conv2D,MaxPooling2D
from tensorflow.python.keras.callbacks import TensorBoard
import pickle
import time

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("Y.pickle","rb")
Y = pickle.load(pickle_in)
print("pickles loaded")

X = X/255.0

dense_layers = [0]
layer_sizes = [256]
conv_layers = [ 3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "CATS_vs_DOGS-{}-conv-{}-nodes-{}-dense-{}-time" \
                   "".format(conv_layer,layer_size,dense_layer,int(time.time()))
            tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

            model = Sequential()

            model.add(Conv2D(64, (3,3) ,input_shape= X.shape[1:]))
            model.add(Activation("relu"))
            model.add(MaxPooling2D(pool_size=(2,2)))

            for l in range(conv_layer-1):
                model.add(Conv2D(64,(3,3)))
                model.add(Activation("relu"))
                model.add(MaxPooling2D(pool_size=(2,2)))

            model.add(Flatten())
            for l in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation("relu"))

            model.add(Dense(1))
            model.add(Activation('sigmoid'))

            model.compile(loss="binary_crossentropy",
                          optimizer="adam",
                          metrics= ["accuracy"])

            model.fit(X,Y, batch_size=32, validation_split=.30,epochs=10,callbacks=[tensorboard])
            model.save("new_model.model")