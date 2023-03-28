

import os
train_normal = os.path.join (r"C:\Users\Isinsu\Desktop\imgs\train2\circle")

train_pneumonia= os.path.join(r"C:\Users\Isinsu\Desktop\imgs\train2\square")


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
    r'C:\Users\Isinsu\Desktop\imgs\train2',
    target_size=(200,200),
    batch_size=32,
    #color_mode = 'grayscale',
    class_mode='binary'
)

validation_generator = datagen.flow_from_directory(
    r'C:\Users\Isinsu\Desktop\imgs\validation2',
    target_size=(200,200),
    batch_size=32,
    #color_mode = 'grayscale', // bu kodu açarsan input shape'deki 3'ü 1 yapmayı unutma.
    class_mode='binary'
)
#from tensorflow.keras.callbacks import EarlyStopping

#early_stopping = EarlyStopping(monitor='val_loss', patience=2)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3),padding="same",
                           activation='linear', input_shape=(200,200, 3)),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.MaxPooling2D((2, 2),padding="same"),
    tf.keras.layers.Dropout(0.25),
    
    tf.keras.layers.Conv2D(32, (3,3),activation='linear',padding="same"),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.MaxPooling2D((2,2),strides=2),
    tf.keras.layers.Dropout(0.25),
       
    tf.keras.layers.Conv2D(64, (3,3),activation='linear',padding="same"),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.MaxPooling2D((2,2),strides=2),
    tf.keras.layers.Dropout(0.4),
    
    
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    
    tf.keras.layers.Dense(128, activation='linear'),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(512, activation='linear'),
    
    tf.keras.layers.Dense(1, activation='sigmoid')
])

from tensorflow.keras.callbacks import EarlyStopping
callbacks=[EarlyStopping(patience=9)]

model.summary()

from tensorflow.keras.optimizers import RMSprop

model.compile(loss="binary_crossentropy",
             optimizer=RMSprop(learning_rate=0.0001),
             metrics=['accuracy'])


history = model.fit(
    train_generator,
    steps_per_epoch=8,
    epochs=50,
    verbose=1,
    validation_data = validation_generator,
    validation_steps = 8,
    callbacks=callbacks
)

import pandas as pd
met_df1 = pd.DataFrame(history.history)
met_df1

met_df1[["accuracy", "val_accuracy"]].plot()
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracies per Epoch")
plt.show()

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