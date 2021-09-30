
# Description : classifies images of cat dog

# import libraries

from keras.models import load_model
from skimage.transform import resize
import tensorflow as tf
from tensorflow import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras import layers

from PIL import ImageTk, Image

import numpy

# from keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

answer = [0, 0, 0, 0, 0]

# Version of style sheet
plt.style.use('fivethirtyeight')

# For resize image

# Load the data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Get the image classification
type = ['airplane', 'automibile', 'bird', 'cat',
        'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# to categorical is function from keras
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

# Normalizes the pixels to be value between 0 and 1
x_train = x_train / 255
x_test = x_test / 255

# Create model
model = Sequential()

# Add first layer
model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(32, 32, 3)))

# Add a pooling layer
model.add(MaxPool2D(pool_size=(2, 2)))

# Add more layer
model.add(Conv2D(32, (5, 5), activation='relu'))

# Add another pooling layer
model.add(MaxPool2D(pool_size=(2, 2)))

# Add a flattening layer
model.add(Flatten())

# Add a layer with 1000 neurons
model.add(Dense(1000, activation='relu'))

# Add a drop out layer
model.add(Dropout(0.5))

# Add a layer with 500 neurons
model.add(Dense(500, activation='relu'))

# Add a drop out layer
model.add(Dropout(0.5))

# Add a layer with 250 neurons
model.add(Dense(250, activation='relu'))

# Add a layer with 10 neurons
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model
hist = model.fit(x_train, y_train_one_hot,
                 batch_size=256,
                 epochs=10,
                 validation_split=0.2)

# Evaluate the model using the test data set
model.evaluate(x_test, y_test_one_hot)[1]


def classification(picture):

    # print(picture)

    # picture = Image.open(picture)

    resized_image = resize(picture, (32, 32, 3))

    predictions = model.predict(np.array([resized_image]))

    # Sort the predictions

    list_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    x = predictions

    max = np.argmax(x[0], axis=0)

    # result = classification[max]

    print(max)

    type2 = type[max]

    return(type2)

    '''

    for i in range(10):
        for j in range(10):
            if x[0][list_index[i]] > x[0][list_index[j]]:
                temp = list_index[i]
                list_index[i] = list_index[j]
                list_index[j] = temp

    for i in range(5):
        answer[i] = (type[list_index[i]], ':', round(
            predictions[0][list_index[i]] * 100, 2), '%')

    return answer





model = load_model('classification_model.h5')

classes = {
    0: 'It\'s a airplane',
    1: 'It\'s a automobile',
    2: 'It\'s a bird',
    3: 'It\'s a cat',
    4: 'It\'s a deer',
    5: 'It\'s a dog',
    6: 'It\'s a frog',
    7: 'It\'s a horse',
    8: 'It\'s a ship',
    9: 'It\'s a truck',
}


def classify(image):

    # image = Image.open(file_path)
    image = Image.resize((32, 32, 3))
    image = numpy.expand_dims(image, axis=0)
    image = numpy.array(image)
    image = image/255

    pred = model.predict_classes([image])[0]
    sign = classes[pred]
    print(sign)

    return(sign)

'''
