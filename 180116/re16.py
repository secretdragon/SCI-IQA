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
from sklearn.feature_extraction.image import extract_patches_2d

#==============================================================================
train_dir='E:/iqa/cnnpy/DTN/'
matfn = 'E:/iqa/cnnpy/score/newD11.mat'
#==============================================================================
 
#data= sio.loadmat(matfn)
#dmos2 = data
def get_files(file_dir,mat_dir,mat_key):
    distorted = []
    for file in os.listdir(file_dir):
        name = file.split(sep = '.')
        distorted.append(file_dir + file)
    
    data=sio.loadmat(mat_dir)
    temp = np.array([])
    dmos2 = data[mat_key]
    dmos2 = dmos2.ravel()
    dmos = np.matrix.tolist(dmos2)
    dmos = dmos[0:47040]+dmos[94840:295764]
#    dmos=dmos[47040:281064]
#    distorted=distorted[47040:281064]
    distorted = distorted[0:47040]+distorted[94840:295764]
    temp = np.array([distorted,dmos])
    temp = temp.transpose()
    np.random.shuffle(temp)
    
    distorted = list(temp[:,0])
    dmos = list(temp[:,1])
    dmos = [float(i) for i in dmos]
    print('There are %d distorted images' %(len(distorted)))
#    print(dmos[1])
#==============================================================================
#     distorted = tf.cast(distorted, tf.string)
#     dmos = tf.cast(dmos, tf.int32)
#     input_queue = tf.train.slice_input_producer([distorted,dmos])
#     image_contents = tf.read_file(input_queue[0])
#     distorted = tf.image.decode_jpeg(image_contents, channels=3)
#     distorted = tf.image.resize_image_with_crop_or_pad(distorted,128,128)
#     distorted = tf.image.per_image_standardization(distorted)
#     dmos = input_queue[1]    
#==============================================================================
    
#    temp = np.array
#    distorted = dis
    return distorted,dmos

def get_test_files(file_dir,mat_dir,mat_dir2,mat_key,mat_key2):
    distorted = []
    for file in os.listdir(file_dir):
        distorted.append(file_dir + file)
    data=sio.loadmat(mat_dir)
    dataw=sio.loadmat(mat_dir2)
    temp = np.array([])
    dmos2 = data[mat_key]
    weight = dataw[mat_key2]

#    dmos=dmos[47040:281064]
#    distorted=distorted[47040:281064]

    dmos2 = dmos2.ravel()
    weight = weight.ravel()
    dmos2 = np.matrix.tolist(dmos2)
    weight = np.matrix.tolist(weight)
#    dmos=dmos2[47040:94080]
#    weight2=weight[47040:94080]
#    distorted=distorted[47040:94080]
    dmos = dmos2[47040:94840]
#    dmos.extend(dmos2[172284:216384])
    weight2 = weight[47040:94840]
#    weight2.extend(weight[172284:216384])
#    distorted = distorted[143472:158172]+distorted[172284:216384]
    distorted = distorted[47040:94840]
    temp = np.array([distorted,dmos,weight2])
    temp = temp.transpose()
#    np.random.shuffle(temp)
    
    distorted = list(temp[:,0])
    dmos = list(temp[:,1])
    weight2=list(temp[:,2])
    dmos = [float(i) for i in dmos]
    weight2=[float(i) for i in weight2]
    print('There are %d distorted images' %(len(distorted)))
#    print(dmos[1])
#==============================================================================
#     distorted = tf.cast(distorted, tf.string)
#     dmos = tf.cast(dmos, tf.int32)
#     input_queue = tf.train.slice_input_producer([distorted,dmos])
#     image_contents = tf.read_file(input_queue[0])
#     distorted = tf.image.decode_jpeg(image_contents, channels=3)
#     distorted = tf.image.resize_image_with_crop_or_pad(distorted,128,128)
#     distorted = tf.image.per_image_standardization(distorted)
#     dmos = input_queue[1]    
#==============================================================================
    
#    temp = np.array
#    distorted = dis
    return distorted,dmos,weight2
#%%    
#==============================================================================
# def get_batch(image, score, image_W, image_H,batch_size, capacity):
#     image = tf.cast(image, tf.string)
#     score = tf.cast(score, tf.int32)
#     
#     input_queue = tf.train.slice_input_producer([image,score])
#     
#     score = input_queue[1]
#     image_contents = tf.read_file(input_queue[0])
#     image = tf.image.decode_jpeg(image_contents, channels=3)
# 
#     image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H) 
#     image = tf.image.per_image_standardization(image)
#     
#     image_batch, score_batch = tf.train.batch([image, score],
#                                               batch_size = batch_size,
#                                               num_threads = 64,
#                                               capacity = capacity)
#     
#     score_batch = tf.reshape(score_batch, [batch_size])
#     
#     return image_batch, score_batch
# 
#==============================================================================

def get_batch(image, score, image_W, image_H,batch_size, capacity):
    image = tf.cast(image, tf.string)
    score = tf.cast(score, tf.float32)
    
#    input_queue = tf.train.slice_input_producer([image,score],shuffle=False)
    input_queue = tf.train.slice_input_producer([image,score])    
    score = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)

    image = tf.image.resize_images(image,(image_W,image_H),method=0) 
    image = tf.image.per_image_standardization(image)

    image_batch, score_batch = tf.train.batch([image, score],
                                              batch_size = batch_size,
                                              num_threads = 64,
                                              capacity = capacity)
    
    score_batch = tf.reshape(score_batch, [batch_size])
    
    return image_batch, score_batch

def get_test_batch(image, score, weight, image_W, image_H,batch_size, capacity):
    image = tf.cast(image, tf.string)
    score = tf.cast(score, tf.float32)
    weight = tf.cast(weight, tf.float32)
    
#    input_queue = tf.train.slice_input_producer([image,score],shuffle=False)
    input_queue = tf.train.slice_input_producer([image,score,weight],shuffle=False)    
    weight = input_queue[2]
    score = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)

    image = tf.image.resize_images(image,(image_W,image_H),method=0) 
    image = tf.image.per_image_standardization(image)

    image_batch, score_batch, weight_batch = tf.train.batch([image, score, weight],
                                              batch_size = batch_size,
                                              num_threads = 64,
                                              capacity = capacity)
    
    score_batch = tf.reshape(score_batch, [batch_size])
    weight_batch = tf.reshape(weight_batch, [batch_size])
    
    return image_batch, score_batch, weight_batch
#%% Test
#import matplotlib.pyplot as plt
#
#BATCH_SIZE = 2
#CAPACITY = 128
#IMG_W = 400
#IMG_H = 400
#    
#imagename,dmos = get_files(train_dir,matfn)
#image_batch, score_batch = get_batch(imagename, dmos, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
#
#with tf.Session() as sess:
#    i = 0
#    coord = tf.train.Coordinator()
#    threads = tf.train.start_queue_runners(coord=coord)
#    
#    try:
#        while not coord.should_stop() and i<1:
#            img, score = sess.run([image_batch, score_batch])
#            
#            for j in np.arange(BATCH_SIZE):
#                print('socre: %d' %score[j])
#                plt.imshow(img[j,:,:,:])
#                plt.show()
#            i+=1
#            
#    except tf.errors.OutOfRangeError:
#        print('done')
#    finally:
#        coord.request_stop()
#    coord.join(threads)
#       