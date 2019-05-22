# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 19:56:47 2018

@author: Icy
"""

# =============================================================================
import tensorflow as tf
#定义两个常量
m1 = tf.constant([[3,3]])
m2 = tf.constant([[2],[3]])
#矩阵乘
produce = tf.matmul(m1,m2)
print(produce)

 #=============================================================================

# =============================================================================
##(1)定义会话，启动默认图
#sess = tf.Session()
#result = sess.run(produce)
#print(result)
#sess.close()
# =============================================================================
# =============================================================================
 #(2)使用with
with tf.Session() as sess:
     result = sess.run(produce)
     print(result)
# =============================================================================
#import tensorflow as tf
##shape：数据形状，选填，默认为value的shape，设置时不得比value小，可以比value阶数、维度更高，超过部分按value提供最后一个数字填充
#constatnValue = tf.constant([1,2,3],shape=[2,4])
#
##创建一个会话
#session = tf.Session()
#print(session.run(constatnValue))
#
##关闭会话
#session.close()
