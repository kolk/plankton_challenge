import numpy as np
import random
import os
import glob
import cv2
import datetime
import pandas as pd
import time
import h5py
import csv

from scipy.misc import imresize, imsave

from sklearn.cross_validation import KFold, train_test_split
from sklearn.metrics import log_loss, confusion_matrix
from sklearn.utils import shuffle

from PIL import Image, ImageChops, ImageOps

#import matplotlib.pyplot as plt

from keras import backend as K
from keras.callbacks import EarlyStopping, Callback
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.optimizers import Adam
from keras.models import Sequential, model_from_json
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, Activation, Dropout, Flatten, Dense, BatchNormalization, MaxPooling2D, Conv2D, LeakyReLU
from keras.callbacks import ModelCheckpoint

train_path = "train/"
model_weights_path = 'model_weights.h5'

n_folds = 2
batch_size = 32
nb_epochs = 50
val_split = 0.15
classes = []
num_classes = 121

img_width, img_height = 80, 80
num_channels = 1

def load_data(train_path):
	X_train = []
	Y_train = []

	print("Loading images......")
	classes = os.listdir(train_path)
	num_classes = len(classes)
	for class_id, directory in enumerate(classes):
		class_path = os.path.join(train_path, directory, '*jpg')
		images = glob.glob(class_path)
		for img in images:
			img = cv2.imread(img).astype('float32')
			img = cv2.resize(img, (img_width, img_height), cv2.INTER_LINEAR)
			img = img / 255.
			img = img.transpose((2, 0, 1)).astype(np.float32)
			X_train.append(img)
			Y_train.append(class_id)
	
	Y_train = np_utils.to_categorical(Y_train, num_classes)
	X_train = np.array(X_train)
	return X_train, Y_train


def build_model():

	model = Sequential()
	
	model.add(Conv2D(32, (7, 7), input_shape = (3, img_width, img_height),  kernel_initializer='he_uniform'))
	BatchNormalization(axis = 1)
	model.add(LeakyReLU())
	model.add(Conv2D(32, (3, 3),  kernel_initializer='he_uniform'))
	BatchNormalization(axis = 1)
	model.add(LeakyReLU())
	model.add(MaxPooling2D(pool_size=(2,2)))

	model.add(Conv2D(64, (3, 3),  kernel_initializer='he_uniform'))
	BatchNormalization(axis=1)
	model.add(LeakyReLU())
	model.add(Conv2D(64, (3, 3),  kernel_initializer='he_uniform'))
	BatchNormalization(axis = 1)
	model.add(LeakyReLU())
	model.add(MaxPooling2D(pool_size=(2,2) ))


	model.add(Conv2D(128, (3, 3),  kernel_initializer='he_uniform'))
	BatchNormalization(axis=1)
	model.add(LeakyReLU())
	model.add(Conv2D(128, (3, 3), kernel_initializer='he_uniform'))
	BatchNormalization(axis = 1)
	model.add(LeakyReLU())
	model.add(MaxPooling2D(pool_size=(2,2)))

		
	model.add(Conv2D(256, (3, 3),  kernel_initializer='he_uniform'))
	BatchNormalization(axis=1)
	model.add(LeakyReLU())
	model.add(Conv2D(256, (3, 3),  kernel_initializer='he_uniform'))
	BatchNormalization(axis = 1)
	model.add(LeakyReLU())
	model.add(MaxPooling2D(pool_size=(2,2)))

	model.add(Flatten())

	model.add(Dense(1024))
	BatchNormalization()
	model.add(LeakyReLU())
	model.add(Dropout(0.4))
	model.add(Dense(121))

	model.add(Activation('softmax'))

	model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-8), metrics=['accuracy'] )
	
	return model

	
def train_model():
	model = build_model()

	train_data, train_labels = load_data(train_path)	
	X_train, X_valid, Y_train, Y_valid = train_test_split(train_data, train_labels, test_size=val_split)
	
	gen = ImageDataGenerator(rotation_range=360, width_shift_range=0.08, shear_range=0.3, height_shift_range=0.08, zoom_range=0.3,  horizontal_flip=True, vertical_flip=True)
	val_generator = ImageDataGenerator()

	train_generator = gen.flow(X_train, Y_train, batch_size=64, shuffle=True)

	test_gen = ImageDataGenerator()
	test_generator = test_gen.flow(X_valid, Y_valid, batch_size=64)


	# checkpoint
	filepath="plankton_weights2/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
	callbacks_list = [checkpoint]

	model.fit_generator(train_generator, steps_per_epoch=25786//64, epochs=3000, validation_data=test_generator, validation_steps=4550//64, callbacks=callbacks_list)	
	
	#model.save_weights(model_weights_path)
	return model


train = False
weights_path = "weights.hdf5"
if train:
        load_data(train_path)
        model = train_model()
else:
        model = build_model()
        model.load_weights(weights_path)
        test_path = '../test'
        test_data = load_test_data(test_path)
        test_prediction = np.argmax(model.predict(test_data, batch_size=64))
        print(test_prediction)

