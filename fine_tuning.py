import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the trained model
trained_model = keras.models.load_model(r"C:\Users\Isinsu\Desktop\imgs\my_cnn_model.h5")

# Freeze the layers of the loaded model
for layer in trained_model.layers:
    layer.trainable = False

# Create a new model for fine-tuning
inputs = keras.Input(shape=(200, 200, 3))
x = trained_model(inputs)
x = layers.Flatten()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
fine_tuned_model = keras.Model(inputs, outputs)

from tensorflow.keras.callbacks import EarlyStopping
callbacks=[EarlyStopping(patience=5)]

# Increase the learning rate
opt = keras.optimizers.RMSprop(learning_rate=1e-3)

# Set more layers to be trainable
for layer in fine_tuned_model.layers[-4:]:
    layer.trainable = True

# Compile the new model
fine_tuned_model.compile(
    loss="binary_crossentropy",
    optimizer=opt,
    metrics=["accuracy"],
)

datagen = ImageDataGenerator()

# Train the new model on your transfer learning dataset
trainForFineTuning = datagen.flow_from_directory(
    r'C:\Users\Isinsu\Desktop\imgs\transferlearning\train',
    target_size=(200,200),
    batch_size=32,
    class_mode='binary'
)

testForFineTuning = datagen.flow_from_directory(
    r'C:\Users\Isinsu\Desktop\imgs\transferlearning\test',
    target_size=(200,200),
    batch_size=32,
    class_mode='binary'
)

# Fine-tune the model
history = fine_tuned_model.fit(
    trainForFineTuning,
    epochs=50,
    validation_data = testForFineTuning,
    callbacks=callbacks
)

# Save the model
fine_tuned_model.save('my_fine_tuned_model.h5')