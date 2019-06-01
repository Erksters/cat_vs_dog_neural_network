import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train,axis=1)
x_test = tf.keras.utils.normalize(x_test,axis=1)

import matplotlib.pyplot as plt

# plt.imshow(x_train[0], cmap= "binary")
# plt.show()

model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))

model.compile(optimizer= 'adam', loss= 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(x_train,y_train,epochs=3)
'''60000/60000 [==============================] - 6s 104us/sample - loss: 0.0723 - acc: 0.9774'''
val_loss, vall_acc = model.evaluate(x_test, y_test)
print(val_loss, "\n", vall_acc)

model.save('epic_num_reader.model')
new_model = tf.keras.models.load_model('epic_num_reader.model')
predictions = new_model.predict(x_test)
print(predictions)

import numpy as np

print(np.argmax(predictions[0]))
plt.imshow(x_test[0])
plt.show()
