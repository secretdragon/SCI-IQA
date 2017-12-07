# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 18:00:06 2017

@author: LENOVO
"""

import os
import numpy as np
import tensorflow as tf
import re16
import mo16

N_CLASSES = 40
IMG_W = 128
IMG_H = 128
BATCH_SIZE = 64
CAPACITY = 2000
MAX_STEP = 20000
#learning_rate = 0.001

#%%
def run_training():
#    train_dir = 'J:/cnnpy/train/'
    train_dir = 'J:/cnnpy/trainpic/'
    matfn = 'J:/cnnpy/score/DMOS16.mat' 
    logs_train_dir = 'J:/cnnpy/newlog/'
    
    train, train_score = re16.get_files(train_dir, matfn)
    
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
    train_logits = mo16.inference(train_batch, BATCH_SIZE,keep_prob=0.5)
    train_loss = mo16.losses(train_logits, train_score_batch)
    train_op = mo16.trainning(train_loss)
    train_acc = mo16.evaluation(train_logits, train_score_batch)
    
    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    saver = tf.train.Saver()
    
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            _, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc])
            if step % 50 == 0:
                print('Step %d, train loss = %.2f, train error = %.2f%%' %(step, tra_loss, tra_acc))
#                print('Learning rate = %2f' %learning_rate)
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str,step)
                
            if step % 2000 == 0 or (step+ 1 ) == MAX_STEP:
                checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
                
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
        
    coord.join(threads)
    sess.close()
    
    #%%
#run_training()
