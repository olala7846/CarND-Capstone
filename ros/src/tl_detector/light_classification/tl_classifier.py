from styx_msgs.msg import TrafficLight

from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import Model, Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

import rospy
import cv2
import random
import numpy as np

import shutil
import os

import tensorflow as tf

IMG_HEIGHT = 299
IMG_WIDTH = 299

def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

def create_data():
    files = os.listdir('src/tl_detector/light_classification/training_data')
    data = [[], [], []]
    for f in files:
        if f.endswith('.jpg'):
            row = f.split('-')
            category = int(row[0])
            datapoint = {'img': 'src/tl_detector/light_classification/training_data/'+f, 'cat': category}
            data[category].append(datapoint)

    # keep the number of green and red lights the same, care less about yellows
    green_rows = len(data[2])

    final_data = []
    for idx in range(3):
        random.shuffle(data[idx])
        min_rows = min(len(data[idx]), green_rows)
        data[idx] = data[idx][:min_rows]
        final_data.extend(data[idx])
    random.shuffle(final_data)

    #validation/training split
    split = int(len(final_data)*.1)
    train_data = final_data[split:]
    validation_data = final_data[:split]
    data = {'train': train_data, 'valid': validation_data}
    basepath = 'src/tl_detector/light_classification'
    for path in data.keys():
        if os.path.exists(os.path.abspath('{}/{}'.format(basepath, path))):
            shutil.rmtree(os.path.abspath('{}/{}'.format(basepath, path)))

        os.makedirs('{}/{}'.format(basepath, path))
        for i in range(3):
            os.makedirs('{}/{}/{}'.format(basepath, path, i))

        for d in data[path]:
            shutil.copy2(d['img'], '{}/{}/{}'.format(basepath, path, d['cat']))

    return len(train_data), len(validation_data)

def create_model(classes):
    if K.image_data_format() == 'channels_first':
        input_shape = (3, IMG_WIDTH, IMG_HEIGHT)
    else:
        input_shape = (IMG_WIDTH, IMG_HEIGHT, 3)
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes))
    model.add(Activation('softmax'))

    return model

def train():
    model = create_model(3)

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    train_samples, valid_samples = create_data()
    print('{} training, {} validation samples'.format(train_samples, valid_samples))

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=5,
        width_shift_range=.05,
        height_shift_range=.05,
        zoom_range=.01,
        horizontal_flip=True)
    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    batch_size = 8
    train_generator = train_datagen.flow_from_directory(
        'src/tl_detector/light_classification/train',
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        'src/tl_detector/light_classification/valid',
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=batch_size,
        class_mode='categorical')

    model.fit_generator(
        train_generator,
        steps_per_epoch=int(1.5*train_samples/batch_size),
        epochs=10,
        validation_data=validation_generator,
        validation_steps=int(valid_samples/batch_size),
        callbacks=[
            ModelCheckpoint(
                'src/tl_detector/model.h5',
                verbose=2,
                save_weights_only=False,
                save_best_only=True
            )
        ])

def train_site():
    model = create_model(4)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=5,
        width_shift_range=.1,
        height_shift_range=.1,
        zoom_range=.05,
        horizontal_flip=True)

    batch_size = 16
    train_generator = train_datagen.flow_from_directory(
            'src/tl_detector/light_classification/site_training_data',
            target_size=(IMG_WIDTH, IMG_HEIGHT),
            batch_size=batch_size,
#            save_to_dir='src/tl_detector/light_classification/aug',
            class_mode='categorical')

    train_samples = 191

    model.fit_generator(
            train_generator,
            steps_per_epoch=int(train_samples/batch_size),
            epochs=20,
            validation_data=train_generator,
            validation_steps=int(train_samples/batch_size),
            callbacks=[
                ModelCheckpoint(
                    'src/tl_detector/site_model.h5',
                    verbose=2,
                    save_weights_only=False,
                    save_best_only=True
                )
            ])


class TLClassifier(object):
    def __init__(self):
        self.model = None
        self.crop_size = IMG_WIDTH

        if rospy.get_param('/launch') == 'styx':
            model_file = 'model.h5'
        else:
            model_file = 'site_model.h5'

        if os.path.isfile(model_file):
            self.model = load_model(model_file)
            self.model._make_predict_function()
            self.graph = tf.get_default_graph()
        else:
            print("Could not load model")


    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if self.model is not None:
            im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            im = im.astype('float32')
            im = preprocess_input(im)
            im_array = np.asarray(im)
            transformed_im_array = im_array[None, :, :, :]
            with self.graph.as_default():
                preds = self.model.predict(transformed_im_array, batch_size=1)
                return np.argmax(preds[0])
        return TrafficLight.UNKNOWN


if __name__ == "__main__":
    train_site()
