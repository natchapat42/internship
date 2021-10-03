from keras.models import load_model
from skimage.transform import resize
import tensorflow as tf
from tensorflow import keras
import glob
import matplotlib.pyplot as plt
from PIL import ImageTk, Image

import numpy
import cv2

# from keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

type = ['airplane', 'automobile', 'bird', 'cat',
        'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

model = load_model('classification_model.h5')

'''

path = glob.glob("C:/Users/march/Desktop/intern/ml/static/uploads/*.jpg")

pictures = path[6]

for i in range(len(path)):

    print(path[i])
'''


def classification(picture):

    #picture = Image.open(picture)

    #image_sequence = picture.getdata()

    #image_array = np.array(image_sequence)

    resized_image = resize(picture, (32, 32, 3))

    print(resized_image.shape)

    predictions = model.predict(np.array([resized_image]))

    print(predictions)

    # Sort the predictions

    list_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    x = predictions

    '''

    for i in range(10):
        for j in range(10):
            if x[0][list_index[i]] > x[0][list_index[j]]:
                temp = list_index[i]
                list_index[i] = list_index[j]
                list_index[j] = temp

    for i in range(5):
        print(type[list_index[i]], ':',
              round(predictions[0][list_index[i]] * 100, 2), '%')

    '''

    max = np.argmax(x[0], axis=0)

    # result = classification[max]

    print(max)

    type2 = type[max]

    return(type2)
