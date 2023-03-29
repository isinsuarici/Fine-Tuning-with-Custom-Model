
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import RMSprop

datagen = ImageDataGenerator()

train_generator = datagen.flow_from_directory(
    r'C:\Users\Isinsu\Desktop\imgs\train',
    target_size=(200,200),
    batch_size=32,
    class_mode='binary'
)

validation_generator = datagen.flow_from_directory(
    r'C:\Users\Isinsu\Desktop\imgs\validation',
    target_size=(200,200),
    batch_size=32,
    class_mode='binary'
)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3),padding="same",
                           activation='relu', input_shape=(200,200, 3)),
    tf.keras.layers.MaxPooling2D((2, 2),strides=2),
    
    tf.keras.layers.Conv2D(32, (3,3),padding="same", activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2),strides=2),
       
    tf.keras.layers.Conv2D(64, (3,3),padding="same", activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2),strides=2),
    
    
    tf.keras.layers.Flatten(),
    
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    
    tf.keras.layers.Dense(1, activation='sigmoid')
])

callbacks=[EarlyStopping(patience=9)]

model.summary()

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

model.save('cnn_model.h5')

met_df1 = pd.DataFrame(history.history)
met_df1[["accuracy", "val_accuracy"]].plot()
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracies per Epoch")
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='best')
plt.show()