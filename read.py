# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 18:10:15 2017

@author: CJN
"""

import tensorflow as tf
import numpy as np
import os
import scipy.io as sio  
import PIL as Image


import six

import cv2 as cv
from sklearn.feature_extraction.image import extract_patches_2d
batch_size = 2

InputImage ='J:/cnnpy/DistortedImages/cim1_1_1.jpg'
InputImage2 = 'J:/cnnpy/DistortedImages/cim2_1_1.jpg'

img = cv.imread(InputImage)
patches = extract_patches_2d(img, (32,32),32)
X = np.transpose(patches.reshape((-1, 32, 32, 3)), (0, 3, 1, 2))
X2 = tf.cast(X,tf.float32)

img2 = cv.imread(InputImage2)
patches = extract_patches_2d(img2, (32,32),32)
X3 = np.transpose(patches.reshape((-1, 32, 32, 3)), (0, 3, 1, 2))
X4 = tf.cast(X3,tf.float32)

image_patches = tf.concat(0,[X2,X4])
image1 = image_patches[0]
image1 = tf.reshape(image1,[32,32,3])
image1 = tf.image.per_image_standardization(image1)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(image1))
