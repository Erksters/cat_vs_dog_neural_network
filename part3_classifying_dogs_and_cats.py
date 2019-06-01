import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten,Conv2D, MaxPooling2D
from tensorflow.python.keras.callbacks import TensorBoard
import pickle
import time

NAME = "Cats-vs-dogs-CNN-With extra-layer"
tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("Y.pickle","rb")
Y = pickle.load(pickle_in)
print("pickles loaded")
X = X/255.0
model = Sequential()
model.add(    Conv2D(64, (3,3) ,input_shape= X.shape[1:])   )
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(64))
model.add(Activation("relu"))


model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics= ["accuracy"])

model.fit(X,Y, batch_size=32, validation_split=.30,epochs=10,callbacks=[tensorboard])


predictions = model.predict(X)


'''22451/22451 [==============================] - 309s 14ms/sample - loss: 0.4986 - acc: 0.7593 - val_loss: 0.4897 - val_acc: 0.7760'''
'''22451/22451 [==============================] - 28s 1ms/sample - loss: 0.4898 - acc: 0.7641 - val_loss: 0.5033 - val_acc: 0.7515'''