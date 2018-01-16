# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 18:00:06 2017

@author: LENOVO
"""

import os
import numpy as np
import tensorflow as tf
import re16
import modelKL

#os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
N_CLASSES = 40
IMG_W = 308
IMG_H = 308
BATCH_SIZE = 32
CAPACITY = 2000
MAX_STEP = 250000
learning_rate = 0.01

#%%
def run_training():
    train_dir = 'E:/iqa/cnnpy/DTN/'
#    train_dir = 'J:/cnnpy/171212/images/'
    matfn = 'E:/iqa/cnnpy/score/newDA.mat' 
#    logs_train_dir = 'J:/cnnpy/171213/'
#    train_dir = 'J:/cnnpy/DTA/'
#    matfn = 'J:/cnnpy/score/newDtrain.mat'  
#    val_dir = 'J:/cnnpy/testpic/'
#    valfn = 'J:/cnnpy/score/DMOS16test.mat'  
#data= sio.loadmat(matfn)
    logs_train_dir = 'G:/newiqa/natural1316'
    log_train_dir = 'G:/newiqa/natural21316'
    
    train, train_score = re16.get_files(train_dir, matfn,'newD')
#    val, val_score = re16.get_files(val_dir, valfn,'DMOS16test')
    train_batch, train_score_batch = re16.get_batch(train,
                                                          train_score,
                                                          IMG_W,
                                                          IMG_H, 
                                                          BATCH_SIZE,
                                                          CAPACITY)
#==============================================================================
#     train_batch, train_score_batch = tf.train.shuffle_batch(
#                                                             [train, train_score],
#                                                             batch_size=BATCH_SIZE,
#                                                             num_threads=64,
#                                                             capacity = CAPACITY,
#                                                             min_after_dequeue = 1000)
#==============================================================================
#    train_logits = mo16.inference(train_batch, BATCH_SIZE,keep_prob=0.5)
    logits = modelKL.inference(train_batch, BATCH_SIZE,keep_prob=0.5)
    loss = modelKL.losses(logits,train_score_batch)
    train_op = modelKL.trainning(loss)
    train_acc = modelKL.evaluation(logits, train_score_batch)

    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(log_train_dir, sess.graph)
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(log_train_dir)
    saver.restore(sess,ckpt.model_checkpoint_path)
#    v_loss = mo16.losses(val_logits, val_score_batch)
#    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#    val_logits = mo16.inference(val_batch,BATCH_SIZE,keep_prob=1)
#    val_loss = mo16.losses(val_logits,val_score_batch)
    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            tra_images,tra_labels = sess.run([train_batch, train_score_batch])
            _, tra_loss, tra_acc = sess.run([train_op, loss, train_acc])
            if step % 50 == 0:
                print('Step %d, train loss = %.2f, train error = %.2f%%' %(step, tra_loss, tra_acc))
#                print('Learning rate = %2f' %learning_rate)
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str,step)
            if step % 2000 == 0 or (step+ 1 ) == MAX_STEP:
                checkpoint_path = os.path.join(log_train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
                
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
        
    coord.join(threads)
    sess.close()
    
    #%%
#run_training()