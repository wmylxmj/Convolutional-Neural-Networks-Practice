# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 13:34:54 2018

@author: wmy
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import math

path = './datasets/'
classes = {'Anime':0, 'Scenery':1, 'Architecture':2, 'Animals':3} 

num_images = 0
num_classes = len(list(enumerate(classes)))
print(num_classes)

for index, name in enumerate(classes):
    class_path = path + name + '\\'
    for img_name in os.listdir(class_path):
        num_images += 1
        pass
    pass

print(num_images)

batch_size = 64

def createdata():
    filename="train.tfrecords"
    writer = tf.python_io.TFRecordWriter(filename) 
    for index, name in enumerate(classes):
        class_path = path + name + '\\'
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name  
            img = Image.open(img_path)
            img = img.resize((256, 256))
            img_raw = img.tobytes() 
            example = tf.train.Example(features=tf.train.Features(feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])), 
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                    }))
            writer.write(example.SerializeToString())  
    writer.close()
    pass

createdata()

print('data created')
    
def read_and_decode(filename, batch_size): 
    filename_queue = tf.train.string_input_producer([filename]) 
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue) 
    features = tf.parse_single_example(serialized_example, 
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })  
 
    img = tf.decode_raw(features['img_raw'], tf.uint8) 
    img = tf.reshape(img, [256, 256, 3])  
    img = tf.cast(img, tf.float32) * (1. / 255)  
    label = tf.cast(features['label'], tf.int64)  
    label = tf.one_hot(label, num_classes)  
    img_batch, label_batch = tf.train.shuffle_batch([img,label],batch_size,500,100, num_threads=16)   
    return img_batch, label_batch


tf.reset_default_graph()

image, label = read_and_decode("train.tfrecords", batch_size)

X = tf.placeholder(tf.float32, [None, 256, 256, 3])
Y = tf.placeholder(tf.int64, [None, num_classes])

tf.set_random_seed(1)   

# layer 1
W1 = tf.get_variable('W1', [3, 3, 3, 32], initializer=tf.contrib.layers.xavier_initializer(seed = 0))
Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
A1 = tf.nn.relu(Z1)
P1 = tf.nn.max_pool(A1, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# layer 2
W2 = tf.get_variable('W2', [1, 1, 32, 32], initializer=tf.contrib.layers.xavier_initializer(seed = 0))
Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding='SAME')
A2 = tf.nn.relu(Z2)

# layer 3
W3 = tf.get_variable('W3', [3, 3, 32, 64], initializer=tf.contrib.layers.xavier_initializer(seed = 0))
Z3 = tf.nn.conv2d(A2, W3, strides=[1, 1, 1, 1], padding='SAME')
A3 = tf.nn.relu(Z3)
P3 = tf.nn.max_pool(A3, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# layer 4
W4 = tf.get_variable('W4', [3, 3, 64, 128], initializer=tf.contrib.layers.xavier_initializer(seed = 0))
Z4 = tf.nn.conv2d(P3, W4, strides=[1, 1, 1, 1], padding='SAME')
A4 = tf.nn.relu(Z4)
P4 = tf.nn.max_pool(A4, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# layer 5
P5 = tf.contrib.layers.flatten(P4)
Z5 = tf.contrib.layers.fully_connected(P5, num_classes, activation_fn=None)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z5,labels=Y))

optimizer = tf.train.AdamOptimizer(0.009).minimize(cost)

init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

predict_op = tf.argmax(Z5, 1)
correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
        
# Calculate accuracy on the test set
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


with tf.Session() as sess:
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for epoch in range(80):
        minibatch_cost = 0.0
        correct = 0
        num_minibatches = int(math.ceil(num_images / batch_size))
        for i in range(num_minibatches):
            x,y=sess.run([image,label])
            feed_dict={X:x , Y:y }
            _, temp_cost = sess.run([optimizer, cost], feed_dict = feed_dict)
            minibatch_cost += temp_cost / num_minibatches
            acc = accuracy.eval({X: x, Y: y})
            corr = int(acc*batch_size)
            correct += corr
            pass
        print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))     
        print("Train Accuracy:", correct/(num_minibatches*batch_size))
        pass
           
    x_test,y_test=sess.run([image,label])
    y_hat = predict_op.eval({X: x_test, Y: y_test}) 
    for index in range(batch_size):
        plt.imshow(x_test[index])
        plt.show()
        if y_hat[index] == 0:
            print('识别为：动漫')
            pass
        elif y_hat[index] == 1:
            print('识别为：风景')
            pass
        elif y_hat[index] == 2:
            print('识别为：建筑')
            pass
        elif y_hat[index] == 3:
            print('识别为：动物')
            pass
        
        pass
        
    coord.request_stop()
    coord.join(threads)




