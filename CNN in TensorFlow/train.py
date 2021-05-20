import tensorflow as tf 
import glob
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop, Adam
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np

BATCH_SIZE = 64
IMAGE_SIZE = 64
EPOCHS = 20

train_dir = os.path.join("dogs-vs-cats/data", 'train')
test_dir = os.path.join("dogs-vs-cats/data", 'test')

## Preprocessing data
# All image will be rescale by 1.0/255, rotate image, zoom, flip, ...
train_datagen = ImageDataGenerator(rescale=1.0/255., rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
                                    zoom_range=0.2, horizontal_flip=True, fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale=1.0/255.)
# Flow from directory training and test image in batch of 32, resize image with shape (64,64)
# Format of data can use flow_from_directory:
# - train_dir
#   - cat
#   - dog  
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(IMAGE_SIZE,IMAGE_SIZE), class_mode='binary', batch_size=BATCH_SIZE)
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(IMAGE_SIZE,IMAGE_SIZE), class_mode='binary', batch_size=BATCH_SIZE)
print(len(train_generator))
## define model
model = tf.keras.models.Sequential([
    # input shape 64x64 with 3 bytes color
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2), 
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'), 
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(), 
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'), 
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('cats') and 1 for the other ('dogs')
    tf.keras.layers.Dense(1, activation='sigmoid')  
])
print(model.summary())

## configure the specifications for model training
# RMSprop automates learning-rate tuning
# Using binary_crossentropy loss, because it's a binary classification problem and our final activation is a sigmoid
model.compile(optimizer=RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics = ['accuracy'])
# Save model with best accuracy
filepath = 'best_acc.h5'
callbacks = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
## Training model with 20 epochs
history = model.fit_generator(train_generator, validation_data=test_generator, steps_per_epoch=len(train_generator), callbacks=[callbacks], epochs=EPOCHS, validation_steps=len(test_generator), workers=6, use_multiprocessing=True)

## Visualize accuracy and loss
# Plot training and validation accuracy per epoch
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("accuracy.png")
plt.show()
# Plot training and validation loss per epoch
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("loss.png")
plt.show()