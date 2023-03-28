

import os
train_normal = os.path.join (r"C:\Users\Isinsu\Desktop\imgs\train\circle")

train_pneumonia= os.path.join(r"C:\Users\Isinsu\Desktop\imgs\train\ellipse")


import matplotlib.image as mpimg
from matplotlib import pyplot as plt
normal_img = [os.path.join(train_normal, file)
              for file in os.listdir(train_normal)[:3]]
plt.figure(figsize=(12, 3))
for i, img_path in enumerate(normal_img):
    sp = plt.subplot(1, 3, i+1)
    sp.axis('Off')
    img = mpimg.imread(img_path)
    plt.imshow(img)
plt.show()

pneumonia_img = [os.path.join(train_pneumonia, file)
              for file in os.listdir(train_pneumonia)[:3]]
plt.figure(figsize=(12, 3))
for i, img_path in enumerate(pneumonia_img):
    sp = plt.subplot(1, 3, i+1)
    sp.axis('Off')
    img = mpimg.imread(img_path)
    plt.imshow(img)
plt.show()

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator()
train_generator = datagen.flow_from_directory(
    r'C:\Users\Isinsu\Desktop\imgs\train',
    target_size=(28, 28),
    batch_size=32,
    color_mode = 'grayscale',
    class_mode='binary'
)

validation_generator = datagen.flow_from_directory(
    r'C:\Users\Isinsu\Desktop\imgs\validation',
    target_size=(28, 28),
    batch_size=32,
    color_mode = 'grayscale',
    class_mode='binary'
)
#from tensorflow.keras.callbacks import EarlyStopping

#early_stopping = EarlyStopping(monitor='val_loss', patience=2)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
       
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    
    tf.keras.layers.Dense(1, activation='sigmoid')
])


model.summary()

from tensorflow.keras.optimizers import RMSprop
model.compile(loss="binary_crossentropy",
             optimizer=RMSprop(learning_rate=0.0001),
             metrics=['accuracy'])


history = model.fit(
    train_generator,
    steps_per_epoch=8,
    epochs=15,
    verbose=1,
    validation_data = validation_generator,
    validation_steps = 8,
    #callbacks=[early_stopping]
)

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracies')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()