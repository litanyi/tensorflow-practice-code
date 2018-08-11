#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import pandas as pd
from  tensorflow.examples.tutorials.mnist import  input_data
from  sklearn.utils import  shuffle
import  tensorflow as tf

from  sklearn import  preprocessing

sess=tf.InteractiveSession()
flags=tf.app.flags
FLAGS=flags.FLAGS

df=pd.read_csv("C:\\Users\litan\Desktop\ile\wine.csv",header=0)
print(df.describe())
X=df[df.columns[1:13]].values
#y=1
y=df['Wine'].values-1
Y=tf.one_hot(indices=y,depth=3,on_value=1.,off_value=0.,axis=1,name="a").eval()

X,Y=shuffle(X,Y)

scaler=preprocessing.StandardScaler()
X=scaler.fit_transform(X)


x=tf.placeholder(tf.float32,[None,12])
W=tf.Variable(tf.zeros([12,3]))
b=tf.Variable(tf.zeros([3]))
y=tf.nn.softmax(tf.matmul(x,W)+b)

y_=tf.placeholder(tf.float32,[None,3])
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_ *tf.log(y),reduction_indices=[1]))
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

tf.initialize_all_variables().run()

for i in range(100):
    X,Y=shuffle(X,Y,random_state=1)
    Xtr=X[0:140,:]
    Ytr=Y[0:140,:]
    Xt=X[140:178,:]
    Yt=Y[140:178,:]
    Xtr,Ytr=shuffle(Xtr,Ytr,random_state=0)
    batch_xs,batch_ys=Xtr,Ytr
    train_step.run({x:batch_xs,y_:batch_ys})
    cost=sess.run(cross_entropy,feed_dict={x:batch_xs,y_:batch_ys})
    correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    print(accuracy.eval({x:Xt,y_:Yt}))






