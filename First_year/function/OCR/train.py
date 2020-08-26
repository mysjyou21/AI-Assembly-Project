from OCR.model import *
import numpy as np
import tensorflow as tf
import os
import time
import argparse
import cv2 as cv


def imshow(img):
    cv.imshow('img', img)
    cv.waitKey(0)
    cv.destroyAllWindows()





def train(args=None):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    imgs = np.load(args.input_img)
    labs = np.load(args.input_label)
    num_imgs = len(imgs)
    num_test = 9
    num_train = num_imgs - num_test
    imgs_train, imgs_test = imgs[num_test:], imgs[:num_test]
    labs_train, labs_test = labs[num_test:], labs[:num_test]

    img_pl = tf.placeholder(tf.float32, shape=[None, 32, 32, 1])
    lab_pl = tf.placeholder(tf.float32, shape=[None, 10])
    is_train = tf.placeholder(tf.bool)
    logit = digit_recognizer(img_pl, is_train)

    with tf.name_scope('train'):
        assert args.loss_type in ['CE', 'MSE', 'focal']
        if args.loss_type == 'CE':
            loss = CE_loss(logit, lab_pl)
        elif args.loss_type == 'MSE':
            loss = MSE_loss(logit, lab_pl)
        else:
            loss = focal_loss(logit, lab_pl, args.gamma)
        acc = tf_get_accuracy(logit, lab_pl)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
            train_op = optimizer.minimize(loss)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

    with tf.Session() as sess:
        sess.run(init_op)
        avg_loss = 0.
        avg_acc = 0.
        step = 0
        num_epochs = args.epoch
        batch_size = args.batch_size
        num_batches = num_train // batch_size
        print('total number of training images:', num_train)
        print('total number of batches:', num_batches)
        print('Start training...')
        current_time = time.time()
        for ep in range(num_epochs):
            idx_list = np.random.permutation(num_train)
            for idx in range(num_batches):
                img_batch = imgs_train[idx_list[idx * batch_size: (idx + 1) * batch_size]]
                lab_batch = labs_train[idx_list[idx * batch_size: (idx + 1) * batch_size]]
                feed_dict = {img_pl: img_batch, lab_pl: lab_batch, is_train: True}
                _, loss_, acc_ = sess.run([train_op, loss, acc], feed_dict=feed_dict)
                avg_loss += loss_ / num_batches
                avg_acc += acc_ / num_batches
                print('Epoch [%02d] [%03d/%03d] Loss: %.5f Acc: %.3f time: %.3f' %
                      (ep+1, idx+1, num_batches, loss_, acc_, time.time() - current_time))

            print('[*] Epoch [%02d] Done' % (ep+1))
            print('[*] Epoch [%03d] Loss: %.5f Acc: %.3f time: %.3f' %
                  (ep+1, avg_loss, avg_acc, time.time() - current_time))

            if not (ep+1) % 1:
                feed_dict = {img_pl: imgs_test, lab_pl: labs_test, is_train: False}
                loss_val, acc_val = sess.run([loss, acc], feed_dict=feed_dict)
                print('[*] Eval loss: %.5f, acc: %.3f\n' % (loss_val, acc_val))
                saver.save(sess,
                           os.path.join(args.weight_path, 'model.ckpt'),
                           global_step=step)
            avg_loss, avg_acc = 0., 0.
            step += 1
    return imgs_train, imgs_test, labs_train, labs_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', default='./weight_mult/')
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--input_img', default='../img_mult.npy')
    parser.add_argument('--input_label', default='../lab_mult.npy')
    parser.add_argument('--loss_type', default='MSE', help='CE for Cross entropy loss, MSE for MSE loss, focal for Focal loss')
    parser.add_argument('--batch_size', default=16)
    parser.add_argument('--lr', default=1e-3)
    parser.add_argument('--epoch', default=10)
    parser.add_argument('--gamma', default=4, help='for focal loss')
    args = parser.parse_args()
    imgs_train, imgs_test, labs_train, labs_test = train(args=args)
