import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Image size for CNN
IMG_SIZE = (224, 224)

def create_datagens(data_dir, batch_size=32):
    """
    Create train, validation, and test generators with augmentation.
    """
    datagen_train = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    datagen_val = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    train_gen = datagen_train.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    val_gen = datagen_val.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    return train_gen, val_gen