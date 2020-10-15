from model import *
import os
import time
import argparse
import numpy as np
import tensorflow as tf

def _get_training(logit, label, args):
    with tf.name_scope('train'):
        assert args.loss_type in ['CE', 'focal']
        if args.loss_type == 'CE':
            loss = CE_loss(logit, label)
        else:
            loss = focal_loss(logit, label, args.gamma)
        acc = tf_get_accuracy(logit, label)
        tf.summary.scalar('loss', loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
        train_op = optimizer.minimize(loss)
    return train_op, loss, acc


def train(args=None):
    tf.compat.v1.reset_default_graph()
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    is_train = True
    original_img = np.load(args.flip_train_img)

    imgs, labs = flip_imgs(original_img)
    num_imgs = len(imgs)
    num_test = int(num_imgs*0.1)
    num_train = num_imgs - num_test
    imgs_train, imgs_test = imgs[num_test:], imgs[:num_test]
    labs_train, labs_test = labs[num_test:], labs[:num_test]

    print('input images shape:', imgs_train.shape)
    print('input labels shape:', labs_train.shape)

    img_pl = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    lab_pl = tf.placeholder(tf.float32, shape=[None, 2])
    logit = flip_recognizer(img_pl, is_train)

    train_op, loss_tensor, acc_tensor = _get_training(logit, lab_pl, args)
    summary_op = tf.summary.merge_all()

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(args.flip_weight_path)
        sess.run(init_op)
        avg_loss = 0.
        avg_acc = 0.
        step = 0
        num_epochs = args.epoch
        batch_size = args.batch_size
        num_batches = num_train // batch_size
        print('Start training...')
        current_time = time.time()
        for ep in range(num_epochs):
            idx_list = np.random.permutation(num_train)
            for idx in range(num_batches):
                img_batch = imgs_train[idx_list[idx * batch_size: (idx + 1) * batch_size]]
                lab_batch = labs_train[idx_list[idx * batch_size: (idx + 1) * batch_size]]
                feed_dict = {img_pl: img_batch, lab_pl: lab_batch}
                _, loss, acc = sess.run([train_op, loss_tensor, acc_tensor], feed_dict=feed_dict)
                avg_loss += loss / num_batches
                avg_acc += acc / num_batches

            print('Epoch [%03d] Loss: %.10f Acc: %.3f time: %.3f' %
                  (ep, avg_loss, avg_acc, time.time() - current_time))

            if not ep % 10:
                summary = sess.run(summary_op, feed_dict=feed_dict)
                train_writer.add_summary(summary, step)

            if not ep % 10:
                feed_dict = {img_pl: imgs_test, lab_pl: labs_test}
                loss_val, acc_val = sess.run([loss_tensor, acc_tensor], feed_dict=feed_dict)
                print('[*] Eval loss: %.5f, acc: %.3f' % (loss_val, acc_val))
                saver.save(sess,
                           os.path.join(args.flip_weight_path, 'model.ckpt'),
                           global_step=step)
            avg_loss, avg_acc = 0., 0.
            step += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--flip_weight_path', default='./flip_weight/')
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--flip_train_img', default='./npy/flip_train_1.npy')
    parser.add_argument('--loss_type', default='focal', help='CE for Cross entropy loss, focal for Focal loss')
    parser.add_argument('--batch_size', default=64)
    parser.add_argument('--lr', default=1e-3)
    parser.add_argument('--epoch', default=100)
    parser.add_argument('--gamma', default=4, help='for focal loss')
    args = parser.parse_args()
    train(args=args)

