import numpy as np
import os
import silence_tensorflow.auto
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from Modules.Plotter import plot_images
from Modules.Con_Mat import plot_confusion_matrix
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPool2D,Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
import warnings
warnings.simplefilter(action='ignore',category=FutureWarning)

physical_devices=tf.config.experimental.list_physical_devices('GPU')
print("Num of GPU's Available ==>",len(physical_devices))



train_path=r'C:\Users\ashut\Machine Learning\Dataset\train'
test_path=r'C:\Users\ashut\Machine Learning\Dataset\test'
valid_path=r'C:\Users\ashut\Machine Learning\Dataset\validation'


train_batches=ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=train_path,target_size=(224,224),classes=['damage','whole'],batch_size=10)
valid_batches=ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=valid_path,target_size=(224,224),classes=['damage','whole'],batch_size=10)
test_batches=ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=test_path,target_size=(224,224),classes=['damage','whole'],batch_size=10,shuffle=False)


imgs,labels=next(train_batches)

plot_images(imgs)

print(labels)

model = Sequential([Conv2D(filters=16,kernel_size=(3,3),activation='relu',padding='same',input_shape=(224,224,3)),MaxPool2D(pool_size=(2,2),strides=2),
                        Conv2D(filters=32,kernel_size=(3,3),activation='relu',padding='same'),MaxPool2D(pool_size=(2,2),strides=2),
                        Dropout(0.60),
                        Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same'),MaxPool2D(pool_size=(2,2),strides=2),
                        Conv2D(filters=128,kernel_size=(3,3),activation='relu',padding='same'),MaxPool2D(pool_size=(2,2),strides=2),
                        Dropout(0.60),
                        Flatten(),
                        Dense(units=2,activation='softmax')])



model.summary()

model.compile(optimizer=Adam(learning_rate=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])

history=model.fit(train_batches,validation_data=valid_batches,epochs=15,verbose=2
,callbacks=[ReduceLROnPlateau(patience=3,factor=0.01)])

print(history.history.keys())

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

test_imgs , test_labels=next(test_batches)
plot_images(test_imgs)
print(test_labels)

test_batches.classes

predictions=model.predict(x=test_batches,verbose=0)
np.round(predictions)


cm=confusion_matrix(y_true=test_batches.classes,y_pred=np.argmax(predictions,axis=-1))


print(test_batches.class_indices)

cm_plot_labels=['damage','whole']

plot_confusion_matrix(cm=cm,classes=cm_plot_labels)

if os.path.isdir(r'C:\Users\ashut\Project\Alpha_small.h5') is False:
        model.save(r'C:\Users\ashut\Project\Alpha_small.h5')