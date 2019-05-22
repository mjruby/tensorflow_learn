# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 17:05:13 2018

@author: Icy
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#使用numpy生成200个随机点
x_data = np.linspace(-0.5,0.5,200)[:,np.newaxis] 
                #np.newaxis 在使用和功能上等价于 None，查看源码发现：newaxis = None，其实就是 None 的一个别名
noise = np.random.normal(0,0.02,x_data.shape)
y_data = np.square(x_data)+noise

#定义两个placeholder
x = tf.placeholder(tf.float32,[None,1])
y = tf.placeholder(tf.float32,[None,1])
        #shape：数据形状。默认是None，就是一维值，也可以是多维，比如[2,3], [None, 3]表示列是3，行不定 

#定义两个神经网络中间层
Weights_L1 = tf.Variable(tf.random.normal([1,10]))
biases_L1 = tf.Variable(tf.zeros([1,10]))
Wx_plus_b_L1 = tf.matmul(x,Weights_L1)+biases_L1#numpy的广播，矩阵扩展运算
L1 = tf.nn.tanh(Wx_plus_b_L1)
        #激励函数

#定义神经网络输出层（只有输入层是没有权重的）
Weights_L2 = tf.Variable(tf.random_normal([10,1]))
biases_L2 = tf.Variable(tf.zeros([1,1]))
Wx_plus_b_L2 = tf.matmul(L1,Weights_L2)+biases_L2
prediction = tf.nn.tanh(Wx_plus_b_L2)

#二次代价函数
loss = tf.reduce_mean(tf.square(y-prediction))
#使用梯度下降法训练
train_step  = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#0.1是步长

with tf.Session() as sess:
    #变量初始化
    sess.run(tf.global_variables_initializer())
    
    for _ in range(2000):
        sess.run(train_step,feed_dict={x:x_data,y:y_data})
        
    #获得预测值
    prediction_value = sess.run(prediction,feed_dict={x:x_data})
    #画图
    plt.figure()  #create a new figure
    plt.scatter(x_data,y_data)  #散点图
    plt.plot(x_data,prediction_value,'r-',lw=5) #红色的 实线，宽度为5
    plt.show()


