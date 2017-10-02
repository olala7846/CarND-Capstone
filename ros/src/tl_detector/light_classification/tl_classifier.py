from styx_msgs.msg import TrafficLight

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.models import model_from_json
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Dense, GlobalAveragePooling2D

import cv2
import random
import numpy as np

import shutil
import os

import tensorflow as tf

IMG_HEIGHT = 299
IMG_WIDTH = 299

class TLClassifier(object):
    def __init__(self):
        self.model = None

        if os.path.isfile('model.h5'):
            self.model = load_model('model.h5')
            self.model._make_predict_function()
            self.graph = tf.get_default_graph()
        else:
            print("Could not load model.json")


    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if self.model is not None:
            datagen = ImageDataGenerator(rescale=1. / 255,)
            im = datagen.standardize(image.astype('float32'))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im_array = np.asarray(im)
            transformed_im_array = im_array[None, :, :, :]
            with self.graph.as_default():
                preds = self.model.predict(transformed_im_array, batch_size=1)
                return np.argmax(preds[0])
        return TrafficLight.UNKNOWN

    def create_data(self):
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
        for path in ['train', 'valid']:
            if os.path.exists(os.path.abspath('{}/{}'.format(basepath, path))):
                shutil.rmtree(os.path.abspath('{}/{}'.format(basepath, path)))

            os.makedirs('{}/{}'.format(basepath, path))
            for i in range(3):
                os.makedirs('{}/{}/{}'.format(basepath, path, i))

            for d in data[path]:
                shutil.copy2(d['img'], '{}/{}/{}'.format(basepath, path, d['cat']))
        return len(train_data), len(validation_data)

    def train(self):
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
        model.add(Dense(3))
        model.add(Activation('sigmoid'))

        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        train_samples, valid_samples = self.create_data()
        print('{} training, {} validation samples'.format(train_samples, valid_samples))

        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            horizontal_flip=True)
        test_datagen = ImageDataGenerator(rescale=1. / 255,)

        batch_size = 16
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
            steps_per_epoch=int(train_samples/batch_size),
            epochs=50,
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


if __name__ == "__main__":
    tl = TLClassifier()
    tl.train()
