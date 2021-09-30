
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras import layers
#from keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

import cv2
import glob

from PIL import Image
import os

from skimage.transform import resize

path = glob.glob("C:/Users/march/Desktop/intern/ml/static/uploads/*.jpg")


for file in path:
    # print(file)
    img = cv2.imread(file)
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


'''
print(len(path))

print(len(path[0]))

print(path[0][47:54])

print(path[0][1])


print(path[1])
print(path[2])


for file in path:

    print(file)

    ''''''

        i = Image.open(file)

        image_sequence = i.getdata()
        image_array = np.array(image_sequence)

        print(image_array)
        print(file)

        os.remove(file)
    '''


# os.remove("C:/Users/march/Desktop/intern/ml/static/uploads/6.jpg")
