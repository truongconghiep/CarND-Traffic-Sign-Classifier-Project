import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Flatten, Dense, Activation, Flatten, Conv2D, MaxPooling2D, Concatenate, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Sequential
from Traffic_Sign_Classifier import *
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.multiclass import unique_labels
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau


class AlexNet_Traffic_Sign_Classifier:
    def __init__(self, batch_size= 100, epochs=100, learn_rate=.001):
        np.random.seed(1000)
        #Defining the parameters
        self.batch_size= 100
        self.epochs=100
        self.learn_rate=.001
        #Instantiation
        self.AlexNet = Sequential()

        #1st Convolutional Layer
        self.AlexNet.add(Conv2D(filters=96, input_shape=(32,32,3), kernel_size=(11,11), strides=(4,4), padding='same'))
        self.AlexNet.add(BatchNormalization())
        self.AlexNet.add(Activation('relu'))
        self.AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

        #2nd Convolutional Layer
        self.AlexNet.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1,1), padding='same'))
        self.AlexNet.add(BatchNormalization())
        self.AlexNet.add(Activation('relu'))
        self.AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

        #3rd Convolutional Layer
        self.AlexNet.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
        self.AlexNet.add(BatchNormalization())
        self.AlexNet.add(Activation('relu'))

        #4th Convolutional Layer
        self.AlexNet.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
        self.AlexNet.add(BatchNormalization())
        self.AlexNet.add(Activation('relu'))

        #5th Convolutional Layer
        self.AlexNet.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
        self.AlexNet.add(BatchNormalization())
        self.AlexNet.add(Activation('relu'))
        self.AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

        #Passing it to a Fully Connected layer
        self.AlexNet.add(Flatten())
        # 1st Fully Connected Layer
        self.AlexNet.add(Dense(4096, input_shape=(32,32,3,)))
        self.AlexNet.add(BatchNormalization())
        self.AlexNet.add(Activation('relu'))
        # Add Dropout to prevent overfitting
        self.AlexNet.add(Dropout(0.4))

        #2nd Fully Connected Layer
        self.AlexNet.add(Dense(4096))
        self.AlexNet.add(BatchNormalization())
        self.AlexNet.add(Activation('relu'))
        #Add Dropout
        self.AlexNet.add(Dropout(0.4))

        #3rd Fully Connected Layer
        self.AlexNet.add(Dense(1000))
        self.AlexNet.add(BatchNormalization())
        self.AlexNet.add(Activation('relu'))
        #Add Dropout
        self.AlexNet.add(Dropout(0.4))

        #Output Layer
        self.AlexNet.add(Dense(43))
        self.AlexNet.add(BatchNormalization())
        self.AlexNet.add(Activation('softmax'))

        #Model Summary
        self.AlexNet.summary()

        self.AlexNet.compile(loss = keras.losses.categorical_crossentropy, optimizer= 'adam', metrics=['accuracy'])

    def training(self, X_train, y_train, X_valid, y_valid, model_file_name='AlexNet_Traffic_Sign_Classifier.h5'):
        #Onehot Encoding the labels.
        y_train=to_categorical(y_train)
        y_val=to_categorical(y_valid)

        #Image Data Augmentation
        train_generator = ImageDataGenerator(rotation_range=2, horizontal_flip=True,zoom_range=.1, data_format='channels_last' )

        val_generator = ImageDataGenerator(rotation_range=2, horizontal_flip=True,zoom_range=.1, data_format='channels_last' )

        #Fitting the augmentation defined above to the data
        train_generator.fit(X_train)
        val_generator.fit(X_valid)

        #Learning Rate Annealer
        lrr= ReduceLROnPlateau(   monitor='val_acc',   factor=.01,   patience=3,  min_lr=1e-5)

        #Training the model
        self.AlexNet.fit_generator(train_generator.flow(X_train, y_train, batch_size=self.batch_size), \
                epochs = self.epochs, steps_per_epoch = X_train.shape[0]//self.batch_size, validation_data = \
                    val_generator.flow(X_valid, y_val, batch_size=self.batch_size), validation_steps = 250, \
                        callbacks = [lrr], verbose=1)

        self.AlexNet.save(model_file_name)