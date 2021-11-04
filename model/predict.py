import silence_tensorflow.auto
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore',category=FutureWarning)


model = tf.keras.models.load_model(r'Programs\Result\Alpha_small.h5')

def prediction(path):
    image_path=path
    img = image.load_img(image_path, target_size=(224, 224, 3))
    img = np.expand_dims(img, axis=0)
    result=model.predict(img)
    print(result)
    if result[0][0] > result[0][1]:
        return "Damaged"
    else:
        return "Not Damaged"

if __name__=="__main__":
    print(prediction(r"C:\Users\ashut\data\test\whole\0018.JPEG"))

