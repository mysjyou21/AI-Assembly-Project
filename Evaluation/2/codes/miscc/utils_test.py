import numpy as np
import glob
import cv2
import os
import tensorflow as tf

def load_classfile(path):
    """ read .cla file
        class file Structure:
            PSB Version_num
            numClasses numModels

            className parentClassName numModelsInClass
            model identifier ...
    """
    file_info = []
    class_idxtonum = {}
    class_numtoidx = {}
    class_instances = {}
    instance_clsnum = {}
    with open(path) as f:
        count = 0
        class_count = 0
        class_num = -1
        for line in f:
            if 'str' in line:
                break
            line = line.rstrip('\n')
            if count <= 1:
                line = line.split(' ')
                file_info += line
            elif len(line)==0 or len(line.split()) == 0:
                continue
            else:
                line = line.split()
                if len(line) > 1: # class info
                    class_num += 1
                    if line[1] == '0':
                        class_idxtonum[line[0]] = [class_num, int(line[1])]  # this-class, parent-class
                    else:
                        parent_class_num = class_idxtonum[line[1]][0]
                        class_idxtonum[line[0]] = [class_num, parent_class_num]
                    class_numtoidx[class_num] = line[0]
                    class_instances[class_num] = []
                else: # model identifier
                    instance_clsnum[int(line[0])] = class_num
                    class_instances[class_num].append(int(line[0]))
            count += 1
        
    print('Read %s file, %s ver.%s, %s classes and %s models' % \
            (path, file_info[0], file_info[1], file_info[2], file_info[3]))

    return class_num+1, int(file_info[3]), class_idxtonum, class_numtoidx, class_instances, instance_clsnum

def load_classfile_str(path):
    """ read .cla file
        class file Structure:
            PSB Version_num
            numClasses numModels

            className parentClassName numModelsInClass
            model identifier ...
    """
    file_info = []
    class_idxtonum = {}
    class_numtoidx = {}
    class_instances = {}
    instance_clsnum = {}
    with open(path) as f:
        count = 0
        class_count = 0
        class_num = -1
        for line in f:
            if 'str' in line:
                break
            line = line.rstrip('\n')
            if count <= 1:
                line = line.split(' ')
                file_info += line
            elif len(line)==0 or len(line.split()) == 0:
                continue
            else:
                line = line.split()
                if len(line) > 1: # class info
                    class_num += 1
                    if line[1] == '0':
                        class_idxtonum[line[0]] = [class_num, int(line[1])]  # this-class, parent-class
                    else:
                        parent_class_num = class_idxtonum[line[1]][0]
                        class_idxtonum[line[0]] = [class_num, parent_class_num]
                    class_numtoidx[class_num] = line[0]
                    class_instances[class_num] = []
                else: # model identifier
                    instance_clsnum[line[0]] = class_num
                    class_instances[class_num].append(line[0])
            count += 1
        
    print('Read %s file, %s ver.%s, %s classes and %s models' % \
            (path, file_info[0], file_info[1], file_info[2], file_info[3]))

    return class_num+1, int(file_info[3]), class_idxtonum, class_numtoidx, class_instances, instance_clsnum


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
    norm_features = tf.math.divide(features, tf.tile(tf.expand_dims(size, 1), (1,features.shape[1])))
    return norm_features

def calculate_distance(z1, z2, same=False):
    """ calculate l2-distance between 2 features' batches z1, z2 
        z1,z2: BxC """
    ######## CHECK #######
    batch_size = tf.shape(z1)[0]

#    dist = tf.matmul(z1, z2, transpose_b=True)

    Z1 = normalize_feature(z1)
    Z2 = normalize_feature(z2)

    Z1 = tf.tile(tf.expand_dims(Z1, 1), [1, batch_size, 1]) # BxBxC
    Z2 = tf.tile(tf.expand_dims(Z2, 0), [batch_size, 1, 1]) # BxBxC

    Z = Z1-Z2 # BxBxC, (Z)ij: = (z1)i - (z2)j

    dist = tf.square(Z)
    dist = tf.reduce_sum(dist, axis=2) # BxB, |(z1)i-(z2)j|^2
    dist = tf.sqrt(dist)

    if same:
        identical = tf.eye(batch_size, num_columns=batch_size) * 100.
        dist += identical

    return dist


def calc_dist(z1, z2):
    """ calculate l2-distance between 2 features' batches z1, z2
        z1: B1xC, z2: B2xC, numpy """
    B1, C = z1.shape
    B2 = z2.shape[0]

#    calc_batch = 32
    
    Z1 = np.tile(np.expand_dims(z1, 1), [1, B2, 1]) # B1xB2xC
    Z2 = np.tile(np.expand_dims(z2, 0), [B1, 1, 1]) # B1xB2xC

    Z = Z1-Z2

    dist = np.square(Z)
    dist = np.sum(dist, axis=2)
    dist = np.sqrt(dist)

    return dist

