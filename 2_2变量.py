# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 20:15:05 2018

@author: Icy
"""

import tensorflow as tf


# =============================================================================
# x = tf.Variable([1,2])
# y = tf.constant([3,3])
#  
# sub = tf.subtract(x,y)
# add = tf.add(x,sub)
#  
#  #有变量时需要个全局变量初始化，否则会报错哦~
# init = tf.global_variables_initializer()
#  
# with tf.Session() as sess:
#     sess.run(init)
#     print(sess.run(sub))
#     print(sess.run(add))
# =============================================================================


#创建一个变量初始化为0 
state = tf.Variable(0,name='counter')
#创建一个op，作用使使state+1
new_value = tf.add(state,1)
update = tf.assign(state,new_value)
 
init = tf.global_variables_initializer()

with tf.Session() as sess:
     sess.run(init)
     print(sess.run(state))
     for each in range(5):
         sess.run(update)
         print(sess.run(state))
         

        