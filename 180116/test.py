# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 23:10:17 2017

@author: LENOVO
"""

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import tensorflow as tf
import re16
import modelKL
import matplotlib.pyplot as plt
import cv2 as cv
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
#==============================================================================
def get_one_image(train, train_score,ind):
    n = len(train)
#    ind = np.random.randint(0,n)
    ind = 1
    img_dir = train[ind]
 
    image = Image.open(img_dir)
    plt.imshow(image)
    score = train_score[ind]
    print(score)
    image = image.resize([128,128])
    image = np.array(image)
    return image,score
#     
def evaluate_one_image():
     
    train_dir = 'E:/iqa/cnnpy/DTA/'
    matfn = 'E:/iqa/cnnpy/score/newDT.mat'
 
    train, train_score = re16.get_files(train_dir, matfn,'newDT')
 #    output = tf.placeholder(tf.float32, shape = [980]) 
 #    for ind in range(len(train)):
    image_array,score = get_one_image(train,train_score,7264)
         
    with tf.Graph().as_default():
        BATCH_SIZE = 1
 
        image = tf.cast(image_array, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image,[1,128,128,3])
 #        score = tf.cast(score, tf.float32)
        logit,conv2 = modelkk.inference(image,BATCH_SIZE,keep_prob=1.0)
         
 #        image = tf.image.resize_image_with_crop_or_pad(image, 320, 320) 
 #        image = tf.image.per_image_standardization(image)
 #        image = tf.reshape(image,[1,320,320,3])        
        x = tf.placeholder(tf.float32, shape = [128,128,3])
 
        logs_train_dir = 'E:/iqa/cnnpy/171229/'
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
            conv1_16 = sess.run(conv2, feed_dict={x:image_array})     # [16, 28, 28 ,1]
            conv1_reshape = sess.run(tf.reshape(conv1_16, [49, 1, 122, 122]))
            fig3,ax3 = plt.subplots(nrows=1, ncols=98, figsize = (49,1))
            for i in range(98):
                ax3[i].imshow(conv1_reshape[i][0])                      # tensor的切片[batch, channels, row, column]
                conv1_reshape[i][0]=(conv1_reshape[i][0]+1)*127.5
                cv.imwrite(('J:/cnnpy/filter/'+str(i)+'.jpg'),conv1_reshape[i][0],[int(cv.IMWRITE_JPEG_QUALITY), 100])
                cv.namedWindow("Image")  
                cv.imshow("Image", conv1_reshape[i][0])  
                cv.waitKey (0) 
            plt.title('Conv1 16x28x28')
            plt.show()

            print('The predicted DMOS of this picture is %.2f' %prediction)
 #            print(score)
 #            print('The ground truth of this picture is %2f' %score)   
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
    
    train_dir = 'J:/cnnpy/DTAT/'
    matfn = 'J:/cnnpy/score/newDT.mat' 

    train, train_score = re16.get_files(train_dir, matfn,'newDT')
    output = np.zeros(64680) 
#    label = np.zeros(100)
    for ind in range(len(output)):
        image_array,score = get_one_image(train,train_score,ind)
        
        with tf.Graph().as_default():
            BATCH_SIZE = 1

            image = tf.cast(image_array, tf.float32)
            image = tf.image.per_image_standardization(image)
            image = tf.reshape(image,[1,128,128,3])
#        score = tf.cast(score, tf.float32)
            logit,conv1 = modelkk.inference(image,BATCH_SIZE,keep_prob=1.0)
        
#        image = tf.image.resize_image_with_crop_or_pad(image, 320, 320) 
#        image = tf.image.per_image_standardization(image)
#        image = tf.reshape(image,[1,320,320,3])        
            x = tf.placeholder(tf.float32, shape = [128,128,3])

            logs_train_dir = 'J:/cnnpy/171219/'
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

    train, train_score = re16.get_files(train_dir, matfn,'newDT')
    output = np.zeros(64680) 
#    label = np.zeros(100)
    for ind in range(len(output)):
        image_array,score = get_one_image(train,train_score,ind)
        
        with tf.Graph().as_default():
            BATCH_SIZE = 1

            image = tf.cast(image_array, tf.float32)
            image = tf.image.per_image_standardization(image)
            image = tf.reshape(image,[1,128,128,3])
#        score = tf.cast(score, tf.float32)
            logit = modelkk.inference(image,BATCH_SIZE,keep_prob=1.0)
        
#        image = tf.image.resize_image_with_crop_or_pad(image, 320, 320) 
#        image = tf.image.per_image_standardization(image)
#        image = tf.reshape(image,[1,320,320,3])        
            x = tf.placeholder(tf.float32, shape = [128,128,3])

            logs_train_dir = 'J:/cnnpy/171219/'
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
        
def evaluate_all_images3():
    output=[] 
    train_dir = 'J:/cnnpy/testpic2/'
    matfn = 'J:/cnnpy/score/DMOS16test.mat' 
    logs_train_dir = 'J:/cnnpy/171216'
    train, train_score = re16.get_files(train_dir, matfn,'DMOS16test')
    train_batch, train_score_batch = re16.get_batch(train,
                                                          train_score,
                                                          128,
                                                          128,
                                                          16,
                                                          2000)
#    output = np.zeros(1568) 
##    label = np.zeros(100)
#    for ind in range(len(output)):
#        image_array,score = get_one_image(train,train_score,ind)
#        
#        with tf.Graph().as_default():
#            BATCH_SIZE = 1
#
#            image = tf.cast(image_array, tf.float32)
#            image = tf.image.per_image_standardization(image)
#            image = tf.reshape(image,[1,128,128,3])
##        score = tf.cast(score, tf.float32)
#            logit = mo16.inference(image,BATCH_SIZE,keep_prob=1.0)
#        
##        image = tf.image.resize_image_with_crop_or_pad(image, 320, 320) 
##        image = tf.image.per_image_standardization(image)
##        image = tf.reshape(image,[1,320,320,3])        
#            x = tf.placeholder(tf.float32, shape = [128,128,3])
#
#            logs_train_dir = 'J:/cnnpy/171216/'
#            saver = tf.train.Saver()
    predict = mo16.inference(train_batch,16,keep_prob=1.0)
    sess = tf.Session()
    image = tf.placeholder(tf.float32, (16, 128, 128, 3))

    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(logs_train_dir)
    saver.restore(sess, ckpt.model_checkpoint_path)
    

    predict = tf.reduce_mean(predict)  
    predict = sess.run(predict, feed_dict={image: train_batch})
    output.append(predict)

#    print(output)
#            print(score)
#            print('The ground truth of this picture is %2f' %score)   
    return output
    
def evaluate_all_images4():
    
    train_dir = 'J:/cnnpy/testpic/'
    matfn = 'J:/cnnpy/score/DMOS16test.mat' 
    im_array=[]
    sc=[]
    train, train_score = re16.get_files(train_dir, matfn,'DMOS16test')
    output = np.zeros(1568) 
#    label = np.zeros(100)
    for ind in range(len(output)):
        image_array,score = get_one_image(train,train_score,ind)
        im_array.append(image_array)
        sc.append(score)
        with tf.Graph().as_default():
            BATCH_SIZE = 1

            image = tf.cast(image_array, tf.float32)
            image = tf.image.per_image_standardization(image)
            image = tf.reshape(image,[1,128,128,3])
#        score = tf.cast(score, tf.float32)
        logit = mo16.inference(image,BATCH_SIZE,keep_prob=1.0)
        
#        image = tf.image.resize_image_with_crop_or_pad(image, 320, 320) 
#        image = tf.image.per_image_standardization(image)
#        image = tf.reshape(image,[1,320,320,3])        
        x = tf.placeholder(tf.float32, shape = [128,128,3])

        logs_train_dir = 'J:/cnnpy/171216/'
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
    
import math
def evaluate():
    with tf.Graph().as_default():
        output=[]
        output2=[]
        output3=[]
#        log_dir = 'C://Users//kevin//Documents//tensorflow//VGG//logsvgg//train//'
        train_dir = 'E:/iqa/cnnpy/DTN/'
        matfn = 'E:/iqa/cnnpy/score/newDA.mat'
        matfn2 = 'E:/iqa/cnnpy/score/weight.mat'
        log_dir = 'G:/newiqa/natural258'
        test, test_score,test_weight = re16.get_test_files(train_dir, matfn,matfn2,'newD','weight',)
        test_batch, test_score_batch, test_weight_batch = re16.get_test_batch(test,
                                                          test_score,
                                                          test_weight,
                                                          308,
                                                          308,
                                                          32,
                                                          3200)
        n_test=len(test)
        logits = modelKL.inference(test_batch,batch_size=32,keep_prob=1.0)
        scores = test_score_batch
        weights = test_weight_batch
#        correct = tools.num_correct_prediction(logits, labels)
#        config=tf.ConfigProto(allow_soft_placement=True)
#        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.65)
#        config.gpu_options.allow_growth=True
#        sess=tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            
            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
                return
        
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess = sess, coord = coord)
            try:
                print('\nEvaluating......')
                num_step = int(math.floor(n_test / 32))
                num_sample = num_step*32
                step = 0
#                total_correct = 0
                while step < num_step and not coord.should_stop():
#                    batch_correct = sess.run(correct)
#                    total_correct += np.sum(batch_correct)
                    out,out2,out3=sess.run([logits,scores,weights])
#                    print(out)
                    output.append(out)
                    output2.append(out2)
                    output3.append(out3)
                    step += 1
                    if step % 50 == 0 :
                        print('step is %d'%step)
#                print('Total testing samples: %d' %num_sample)
#                print('Total correct predictions: %d' %total_correct)
#                print('Average accuracy: %.2f%%' %(100*total_correct/num_sample))
            except Exception as e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
                coord.join(threads)
        output=np.array(output)
        sio.savemat('outputn1316_2.mat',{'data':output,'score':output2,'weight':output3})
    return output
                
#%%    


