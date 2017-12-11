# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 23:10:17 2017

@author: LENOVO
"""

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import re16
import mo16

#==============================================================================
# def get_one_image(train, train_score,ind):
#     n = len(train)
# #    ind = np.random.randint(0,n)
# #    ind = 1
#     img_dir = train[ind]
# 
#     image = Image.open(img_dir)
#     plt.imshow(image)
#     score = train_score[ind]
#     print(score)
#     image = image.resize([32,32])
#     image = np.array(image)
#     return image,score
#     
# def evaluate_one_image():
#     
#     train_dir = 'J:/segmentation/testtext'
#     matfn = 'J:/segmentation/textDtest.mat'  
# 
#     train, train_score = nr.get_files(train_dir, matfn)
# #    output = tf.placeholder(tf.float32, shape = [980]) 
# #    for ind in range(len(train)):
#     image_array,score = get_one_image(train,train_score)
#         
#     with tf.Graph().as_default():
#         BATCH_SIZE = 1
#         N_CLASSES = 40
# 
#         image = tf.cast(image_array, tf.float32)
#         image = tf.image.per_image_standardization(image)
#         image = tf.reshape(image,[1,32,32,3])
# #        score = tf.cast(score, tf.float32)
#         logit = modelKL.inference(image,BATCH_SIZE)
#         
# #        image = tf.image.resize_image_with_crop_or_pad(image, 320, 320) 
# #        image = tf.image.per_image_standardization(image)
# #        image = tf.reshape(image,[1,320,320,3])        
#         x = tf.placeholder(tf.float32, shape = [32,32,3])
# 
#         logs_train_dir = 'J:/cnnpy/logs/train/'
#         saver = tf.train.Saver()
#         
#         with tf.Session() as sess:
#             
#             print('Reading checkpoints...')
#             ckpt = tf.train.get_checkpoint_state(logs_train_dir)
#             if ckpt and ckpt.model_checkpoint_path:
#                 global_step = ckpt.model_checkpoint_path.split('/')[-1].split('.')[-1]
#                 saver.restore(sess, ckpt.model_checkpoint_path)
#                 print('Loading success, global_step is %s' % global_step)
#             else:
#                 print('No checkpoint file found')
#                 
#             prediction = sess.run(logit, feed_dict ={x: image_array})
#             print('The predicted DMOS of this picture is %.2f' %prediction)
# #            print(score)
# #            print('The ground truth of this picture is %2f' %score)   
#     
# def evaluate_all_images():
#     
#     train_dir = 'J:/segmentation/testtext/'
#     matfn = 'J:/segmentation/textDtest.mat' 
# 
#     train, train_score = nr.get_files(train_dir, matfn)
#     output = np.zeros(38906) 
# #    label = np.zeros(100)
#     for ind in range(len(output)):
#         image_array,score = get_one_image(train,train_score,ind)
#         
#         with tf.Graph().as_default():
#             BATCH_SIZE = 1
#             N_CLASSES = 40
# 
#             image = tf.cast(image_array, tf.float32)
#             image = tf.image.per_image_standardization(image)
#             image = tf.reshape(image,[1,32,32,3])
# #        score = tf.cast(score, tf.float32)
#             logit = modelKL.inference(image,BATCH_SIZE,keep_prob=1.0)
#         
# #        image = tf.image.resize_image_with_crop_or_pad(image, 320, 320) 
# #        image = tf.image.per_image_standardization(image)
# #        image = tf.reshape(image,[1,320,320,3])        
#             x = tf.placeholder(tf.float32, shape = [32,32,3])
# 
#             logs_train_dir = 'J:/segmentation/textlog/'
#             saver = tf.train.Saver()
#         
#             with tf.Session() as sess:
#             
#                 print('Reading checkpoints...')
#                 ckpt = tf.train.get_checkpoint_state(logs_train_dir)
#                 if ckpt and ckpt.model_checkpoint_path:
#                     global_step = ckpt.model_checkpoint_path.split('/')[-1].split('.')[-1]
#                     saver.restore(sess, ckpt.model_checkpoint_path)
#                     print('Loading success, global_step is %s' % global_step)
#                 else:
#                     print('No checkpoint file found')
#                 
#                 prediction = sess.run(logit, feed_dict ={x: image_array})
#                 print('ind = %f The predicted DMOS of this picture is %.2f' %(ind, prediction))
#                 output[ind] = prediction 
# 
#     print(output)
# #            print(score)
# #            print('The ground truth of this picture is %2f' %score)   
#     return output
# 
# def evaluate_all_images2():
#     
#     train_dir = 'J:/cnnpy/DTAT/'
#     matfn = 'J:/cnnpy/score/newDT.mat'  
# 
#     train, train_score = nr.get_files(train_dir, matfn)
#     output = np.zeros(64680) 
# #    label = np.zeros(100)
# 
#         
#     with tf.Graph().as_default():
#         BATCH_SIZE = 32
#         image_array = tf.cast(train, tf.string)
#         score = tf.cast(train_score, tf.int32)
#     
#         input_queue = tf.train.slice_input_producer([image_array,score])
#     
#         score = input_queue[1]
#         image_contents = tf.read_file(input_queue[0])
#         image = tf.image.decode_jpeg(image_contents, channels=3)
# 
#         image = tf.image.resize_images(image,(128,128),method=0) 
#         image = tf.image.per_image_standardization(image)
#         image = tf.cast(image_array, tf.float32)
# #        image = tf.image.per_image_standardization(image)
# #        image = tf.reshape(image,[1,128,128,3])
# ##        score = tf.cast(score, tf.float32)
#         logit = modelKL.inference(image,BATCH_SIZE,keep_prob=1.0)
# 
#     
#         image_batch, score_batch = tf.train.batch([image, score],
#                                               batch_size = BATCH_SIZE,
#                                               num_threads = 64,
#                                               capacity = 2000)
#     
#         score = tf.reshape(score_batch, [BATCH_SIZE])
# #        image = tf.image.resize_image_with_crop_or_pad(image, 320, 320) 
# #        image = tf.image.per_image_standardization(image)
# #        image = tf.reshape(image,[1,320,320,3])        
#         x = tf.placeholder(tf.float32, shape = [128,128,3])
# 
#         logs_train_dir = 'J:/segmentation/textlog/'
#         saver = tf.train.Saver()
#         
#         with tf.Session() as sess:
#             
#             print('Reading checkpoints...')
#             ckpt = tf.train.get_checkpoint_state(logs_train_dir)
#             if ckpt and ckpt.model_checkpoint_path:
#                 global_step = ckpt.model_checkpoint_path.split('/')[-1].split('.')[-1]
#                 saver.restore(sess, ckpt.model_checkpoint_path)
#                 print('Loading success, global_step is %s' % global_step)
#             else:
#                 print('No checkpoint file found')
#                 
#             prediction = sess.run(logit, feed_dict ={x: image_batch})
#             print('ind = %f The predicted DMOS of this picture is %.2f' %(ind, prediction))
#             output[ind] = prediction 
# 
#     print(output)
# #            print(score)
# #            print('The ground truth of this picture is %2f' %score)   
#     return output
#==============================================================================
def get_one_image(train, train_score,ind):
    n = len(train)
#    ind = np.random.randint(0,n)
#    ind = 1
    img_dir = train[ind]

    image = Image.open(img_dir)
    plt.imshow(image)
    score = train_score[ind]
    print(score)
    image = image.resize([128,128])
    image = np.array(image)
    return image,score
    
def evaluate_all_images():
    
    train_dir = 'J:/cnnpy/testpic/'
    matfn = 'J:/cnnpy/score/DMOS16test.mat' 

    train, train_score = re16.get_files(train_dir, matfn)
    output = np.zeros(1568) 
#    label = np.zeros(100)
    for ind in range(len(output)):
        image_array,score = get_one_image(train,train_score,ind)
        
        with tf.Graph().as_default():
            BATCH_SIZE = 1
            N_CLASSES = 40

            image = tf.cast(image_array, tf.float32)
            image = tf.image.per_image_standardization(image)
            image = tf.reshape(image,[1,128,128,3])
#        score = tf.cast(score, tf.float32)
            logit = mo16.inference(image,BATCH_SIZE,keep_prob=1.0)
        
#        image = tf.image.resize_image_with_crop_or_pad(image, 320, 320) 
#        image = tf.image.per_image_standardization(image)
#        image = tf.reshape(image,[1,320,320,3])        
            x = tf.placeholder(tf.float32, shape = [128,128,3])

            logs_train_dir = 'J:/cnnpy/171211/'
            saver = tf.train.Saver()
        
            with tf.Session() as sess:
            
                print('Reading checkpoints...')
                ckpt = tf.train.get_checkpoint_state(logs_train_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('.')[-1]
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print('Loading success, global_step is %s' % global_step)
                else:
                    print('No checkpoint file found')
                
                prediction = sess.run(logit, feed_dict ={x: image_array})
                print('ind = %f The predicted DMOS of this picture is %.2f' %(ind, prediction))
                output[ind] = prediction 

    print(output)
#            print(score)
#            print('The ground truth of this picture is %2f' %score)   
    return output

def evaluate_all_images2():
    
    train_dir = 'J:/cnnpy/DTAT/'
    matfn = 'J:/cnnpy/score/newDT.mat'  

    train, train_score = nr.get_files(train_dir, matfn)
    output = np.zeros(64680) 
#    label = np.zeros(100)

        
    with tf.Graph().as_default():
        BATCH_SIZE = 32
        image_array = tf.cast(train, tf.string)
        score = tf.cast(train_score, tf.int32)
    
        input_queue = tf.train.slice_input_producer([image_array,score])
    
        score = input_queue[1]
        image_contents = tf.read_file(input_queue[0])
        image = tf.image.decode_jpeg(image_contents, channels=3)

        image = tf.image.resize_images(image,(128,128),method=0) 
        image = tf.image.per_image_standardization(image)
        image = tf.cast(image_array, tf.float32)
#        image = tf.image.per_image_standardization(image)
#        image = tf.reshape(image,[1,128,128,3])
##        score = tf.cast(score, tf.float32)
        logit = modelKL.inference(image,BATCH_SIZE,keep_prob=1.0)

    
        image_batch, score_batch = tf.train.batch([image, score],
                                              batch_size = BATCH_SIZE,
                                              num_threads = 64,
                                              capacity = 2000)
    
        score = tf.reshape(score_batch, [BATCH_SIZE])
#        image = tf.image.resize_image_with_crop_or_pad(image, 320, 320) 
#        image = tf.image.per_image_standardization(image)
#        image = tf.reshape(image,[1,320,320,3])        
        x = tf.placeholder(tf.float32, shape = [128,128,3])

        logs_train_dir = 'J:/segmentation/textlog/'
        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            
            print('Reading checkpoints...')
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('.')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
                
            prediction = sess.run(logit, feed_dict ={x: image_batch})
            print('ind = %f The predicted DMOS of this picture is %.2f' %(ind, prediction))
            output[ind] = prediction 

    print(output)
#            print(score)
#            print('The ground truth of this picture is %2f' %score)   
    return output
        
