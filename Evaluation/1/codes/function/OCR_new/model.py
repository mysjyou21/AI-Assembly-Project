from ops import *
import cv2
from random import *

def digit_recognizer(x, is_train, reuse=False):
    """Simplest model for assembly number classification"""
    # input: [batch_size, None, None, 1]\
    # architecture: 512 fc - 512 fc - 10 fc
    with tf.variable_scope('model2', reuse=reuse):
        x = flatten(x)
        out = mlp(x, 784, 'fc1_', is_train, reuse, norm=None, activation='relu')
        out = mlp(out, 784, 'fc2_', is_train, reuse, norm=None, activation='relu')
        out = mlp(out, 10, 'fc3_', is_train, reuse, norm=None, activation=None)
    return out

def flip_recognizer(x, is_train, reuse=False):
    """Simplest model for assembly number classification"""
    # input: [batch_size, None, None, 1]\
    # architecture: 512 fc - 512 fc - 10 fc
    with tf.variable_scope('model1', reuse=reuse):
        x = flatten(x)
        out = mlp(x, 512, 'fc1', is_train, reuse, norm=None, activation='relu')
        out = mlp(out, 512, 'fc2', is_train, reuse, norm=None, activation='relu')
        out = mlp(out, 256, 'fc3', is_train, reuse, norm=None, activation='relu')
        out = mlp(out, 256, 'fc4', is_train, reuse, norm=None, activation='relu')
        out = mlp(out, 2, 'fc5', is_train, reuse, norm=None, activation=None)
    return out


def CE_loss(logit, label):
    loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(label, logit))
    return loss


def focal_loss(logit, label, gamma):
    logit = tf.nn.softmax(logit)
    eps = 1e-9
    weight = tf.pow(1. - logit, gamma)
    loss = -tf.reduce_sum(weight * label * tf.log(logit + eps))
    return loss


def tf_get_accuracy(logit, label):
    logit = tf.nn.softmax(logit)
    pred = tf.argmax(logit, axis=1)
    ans = tf.argmax(label, axis=1)
    equal = tf.cast(tf.equal(pred, ans), dtype=tf.float32)
    acc = tf.reduce_mean(equal)
    return acc

def tf_report_ans(logit, label):
    logit = tf.nn.softmax(logit)
    pred = tf.argmax(logit, axis=1)
    ans = tf.argmax(label, axis=1)
    return pred, ans

def tf_report_pred(logit):
    logit = tf.nn.softmax(logit)
    pred = tf.argmax(logit, axis=1)
    return pred

def flip_imgs(imgs):
    flip_index = []
    labs = np.zeros([len(imgs), 2])
    labs[:, 0] = 1
    while flip_index.__len__() < len(imgs)*0.2:
        f = randint(0, len(imgs)-1)
        if f not in flip_index:
            flip_index.append(f)

    print("flipped img : ", len(flip_index))
    for i in range(len(flip_index)):
        ori = imgs[flip_index[i]]
        flip = cv2.flip(ori, -1)
        imgs[flip_index[i]] = flip[:, :, np.newaxis]
        labs[flip_index[i]] = [0, 1]
    return imgs, labs

def get_indexed_ones(index, img=None):
    indexed_img = []
    for i in range(len(index)):
        if img is not None:
            indexed_img.append(img[index[i]])
    indexed_img = np.array(indexed_img)
    return indexed_img


def flipping_words(imgs):
    imgs_list = []
    num_imgs = len(imgs)
    for i in range(num_imgs):
        img = imgs[num_imgs - i - 1]
        flipped_img = np.rot90(img, 2)
        imgs_list.append(flipped_img)
    flipped_imgs = np.reshape(imgs_list, (-1, 28, 28, 1))
    return flipped_imgs

