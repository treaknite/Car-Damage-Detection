import silence_tensorflow.auto
import tensorflow as tf
import os

if os.path.isdir(r'C:\Users\ashut\Project\Alpha_small.tflite') is False:
    model = tf.keras.models.load_model(r'C:\Users\ashut\Project\Alpha_small.h5')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    open("Alpha_small.tflite", "wb").write(tflite_model)

else:
    print("File is already present in the folder!!")