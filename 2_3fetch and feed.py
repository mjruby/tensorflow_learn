# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 20:34:49 2018

@author: Icy
"""

import tensorflow as tf
#Fetch--可同时run多个op
input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)

add = tf.add(input2,input3)
mul = tf.multiply(input1,add)

with tf.Session() as sess:
    result = sess.run([mul,add])
    print('Fetch形式：',result)
    
#Feed
#创建占位符
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1,input2)

with tf.Session() as sess:
    #feed的数据以字典的形式传入
    print('Feed形式：',sess.run(output,feed_dict={input1:[7.0],input2:[2.0]}))
    
    
    