import cv2
import tensorflow as tf

Categories = ["Dog" , "Cat"]

def prepare(filepath):
    IMG_SIZE = 30
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
    return new_array.reshape(-1,IMG_SIZE,IMG_SIZE,1)

model = tf.keras.models.load_model("new_model.model")

prediction = model.predict([prepare("cat2.jpg")])
print(Categories[int(prediction[0][0])])
print(int(prediction[0][0]))