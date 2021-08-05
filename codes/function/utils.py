import numpy as np
import glob
import cv2
import os
import tensorflow as tf

################## DETECTION ##############################


################## RETRIEVAL ##############################

def transform(img, size=None):
    """ transform UTF-8 image """
    # BGR to RGB --when img is a color image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # resize
    if size is not None:
        img = cv2.resize(img, size)
    
    # normalize
    img = (img).astype(np.float32)/255.

    return img

def normalize_feature(features):
    ep=0 #1e-10
    size = tf.sqrt(tf.reduce_sum(tf.square(features), axis=1)+ep)
    norm_features = tf.divide(features, tf.tile(tf.expand_dims(size, 1), (1,features.shape[1])))
    return norm_features

def calculate_distance(z1, z2, same=False):
    """ calculate l2-distance between 2 features' batches z1, z2 
        z1,z2: B1xC , B2XC """
    ######## CHECK #######
    batch_size1 = tf.shape(z1)[0] #z1 = batch_size1 X C
    batch_size2 = tf.shape(z2)[0] #z2 = batch_size2 X C

    Z1 = normalize_feature(z1)
    Z2 = normalize_feature(z2)

    Z1 = tf.tile(tf.expand_dims(Z1, 1), [1, batch_size2, 1]) # B1xB2xC
    Z2 = tf.tile(tf.expand_dims(Z2, 0), [batch_size1, 1, 1]) # B1xB2xC

    Z = Z1-Z2 # BxBxC, (Z)ij: = (z1)i - (z2)j

    dist = tf.square(Z)
    dist = tf.reduce_sum(dist, axis=2) # BxB, |(z1)i-(z2)j|^2
    dist = tf.sqrt(dist)

    if same:
        identical = tf.eye(batch_size, num_columns=batch_size) * 100.
        dist += identical

    return dist


############################# POSE ################################################

def print_time(name, time):
	print(name + ' time : {} hr {} min {} sec'.format(int(time/60/60), int(time/60%60), int(time%60)))
