from styx_msgs.msg import TrafficLight

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.models import model_from_json
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Dense, GlobalAveragePooling2D

import csv
import random
import numpy as np

import shutil
import os

IMG_HEIGHT = 299
IMG_WIDTH = 299

class TLClassifier(object):
    def __init__(self):
        if os.path.isfile('model.json'):
            with open('model.json', 'r') as jfile:
                model = model_from_json(jfile.read())

            model.compile("adam", "mse")
            weights_file = args.model.replace('json', 'h5')
            model.load_weights(weights_file)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        return TrafficLight.UNKNOWN

    def create_data(self):
        with open("src/tl_detector/light_classification/images.csv") as f:
            reader = csv.reader(f)
            rows = list(reader)
        data = [[], [], []]
        for row in rows:
            category = int(row[1])
            datapoint = {'img': row[0], 'cat': category}
            data[category].append(datapoint)
        min_rows = min(len(data[0]), min(len(data[2]), len(data[1])))
        print(min_rows)
        final_data = []
        for idx in range(3):
            random.shuffle(data[idx])
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

    def train(self):
        if K.image_data_format() == 'channels_first':
            input_shape = (3, 256, 256)
        else:
            input_shape = (256, 256, 3)
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

        self.create_data()
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            horizontal_flip=True)
        test_datagen = ImageDataGenerator(rescale=1. / 255,)

        train_generator = train_datagen.flow_from_directory(
            'src/tl_detector/light_classification/train',
            batch_size=32,
            class_mode='categorical')

        validation_generator = test_datagen.flow_from_directory(
            'src/tl_detector/light_classification/valid',
            batch_size=32,
            class_mode='categorical')

        json_string = model.to_json()
        with open('model.json', mode='w') as outfile:
            outfile.write(json_string)

        model.fit_generator(
            train_generator,
            steps_per_epoch=2000,
            epochs=50,
            validation_data=validation_generator,
            validation_steps=800,
            callbacks=[
                ModelCheckpoint(
                    'model.h5',
                    verbose=2,
                    save_weights_only=True,
                    save_best_only=True
                )
            ])


if __name__ == "__main__":
    tl = TLClassifier()
    tl.train()
