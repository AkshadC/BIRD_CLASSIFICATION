from keras.src.preprocessing.image import ImageDataGenerator
import os
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow import keras
from tensorflow.keras import models
import cv2


class InceptionV3Model:
    def __init__(self, train_dir, test_dir, BATCH_SIZE, IMAGE_SHAPE):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.BATCH_SIZE = BATCH_SIZE
        self.IMAGE_SHAPE = (IMAGE_SHAPE, IMAGE_SHAPE)
        self.model = None

    def create_train_data_generators(self, rescale: bool = True, shear_range=0.5, zoom_range=0.5, width_shift_range=0.5,
                                     height_shift_range=0.5, horizontal_flip: bool = True, class_mode='categorical'):
        """

        :param rescale: Default is True
        :param shear_range: Default value 0.5
        :param zoom_range: Default value 0.5
        :param width_shift_range: Default value 0.5
        :param height_shift_range: Default value 0.5
        :param horizontal_flip: Amount of flip you want you're images, default = 0.5
        :param class_mode: Can be "categorical", "binary", "sparse", "input"
        :return: Returns a pair of train and test data generator
        """
        if rescale:
            train_datagen = ImageDataGenerator(
                rescale=1. / 255,
                shear_range=shear_range,
                zoom_range=zoom_range,
                width_shift_range=width_shift_range,
                height_shift_range=height_shift_range,
                horizontal_flip=horizontal_flip)
            test_datagen = ImageDataGenerator(rescale=1. / 255)
        else:
            train_datagen = ImageDataGenerator(
                shear_range=shear_range,
                zoom_range=zoom_range,
                width_shift_range=width_shift_range,
                height_shift_range=height_shift_range,
                horizontal_flip=horizontal_flip)
            test_datagen = ImageDataGenerator()

        train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=self.IMAGE_SHAPE,
            batch_size=self.BATCH_SIZE,
            class_mode=class_mode)

        validation_generator = test_datagen.flow_from_directory(
            self.test_dir,
            target_size=self.IMAGE_SHAPE,
            batch_size=self.BATCH_SIZE,
            class_mode=class_mode)
        return train_generator, validation_generator

    def create_model(self, best_save_model_path="Models/"):
        inception = InceptionV3(weights='imagenet', include_top=False)
        x = inception.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)

        predictions = Dense(68, kernel_regularizer=regularizers.l2(0.005), activation='softmax')(x)
        best_save_model_path = best_save_model_path + 'model_inceptionV3.hdf'
        model = Model(inputs=inception.input, outputs=predictions)
        model.compile(optimizer=SGD(lr=0.0001, momentum=0.75), loss='categorical_crossentropy', metrics=['accuracy'])
        checkpointer = ModelCheckpoint(filepath=best_save_model_path, verbose=1, save_best_only=True)
        csv_logger = CSVLogger('history_3class.log')
        self.model = model
