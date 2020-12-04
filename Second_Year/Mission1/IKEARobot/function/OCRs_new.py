from function.utilities.utils import *
from function.OCR_new.model import *


class OCRModel():
    def __init__(self, args):
        with args.graph_OCR.as_default():
            sess = args.sess_OCR
            self.img_pl = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
            self.logit_class = digit_recognizer(self.img_pl, is_train=False, reuse=tf.AUTO_REUSE)
            self.logit_flip = flip_recognizer(self.img_pl, is_train=False, reuse=tf.AUTO_REUSE)
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            var_list_class = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model2')
            var_list_flip = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model1')
            saver_class = tf.train.Saver(var_list=var_list_class)
            saver_flip = tf.train.Saver(var_list=var_list_flip)
            print('OCR MODEL : Loading weights from %s' % tf.train.latest_checkpoint(args.opt.ocr_class_weight_path))
            saver_class.restore(sess, tf.train.latest_checkpoint(args.opt.ocr_class_weight_path))
            print('OCR MODEL : Loading weights from %s' % tf.train.latest_checkpoint(args.opt.ocr_flip_weight_path))
            saver_flip.restore(sess, tf.train.latest_checkpoint(args.opt.ocr_flip_weight_path))

    def run_OCR_serial(self, args, imgs):
        # 1. imgs를 (-1, 28, 28, 1)로 맞추기
        new_imgs = []
        for img in imgs:
            img = img[3:31, 3:31]
            new_imgs.append(img)
        new_imgs = np.reshape(new_imgs, (-1, 28, 28, 1))

        with args.graph_OCR.as_default():
            sess = args.sess_OCR
            index_1 = self.classifying(session=sess, imgs=new_imgs, res=1)
            is_flipped = self.is_flipped(session=sess, imgs=new_imgs, index=index_1)
            if is_flipped:
                flipped_imgs = self.flipping_words(imgs=new_imgs)
                final_index = self.classifying(session=sess, imgs=flipped_imgs)
            else:
                final_index = self.classifying(session=sess, imgs=new_imgs)
            result_string = ''
            for num in final_index:
                result_string += str(num)
            return result_string

    def run_OCR_mult(self, args, imgs):
        new_imgs = []
        for img in imgs:
            img = img[3:31, 3:31]
            new_imgs.append(img)
        new_imgs = np.reshape(new_imgs, (-1, 28, 28, 1))

        with args.graph_OCR.as_default():
            sess = args.sess_OCR
            final_index = self.classifying(session=sess, imgs=new_imgs)
            result_string = ''
            for num in final_index:
                result_string += str(num)
            return result_string


    def classifying(self, session, imgs, res=None):
        index = []
        pred = tf.argmax(tf.nn.softmax(self.logit_class), axis=1)
        feed_dict_test = {self.img_pl: imgs}
        pred_ = session.run(pred, feed_dict=feed_dict_test)

        if res is not None:
            for i in range(len(pred_)):
                if pred_[i] == 1:
                    index.append(i)
        else:
            index = pred_
        return index

    def is_flipped(self, session, imgs, index=None):
        imgs = get_indexed_ones(index, imgs)
        if len(imgs) == 0:
            return False
        pred = tf.argmax(tf.nn.softmax(self.logit_flip), axis=1)
        flipped_index = []

        feed_dict_flip = {self.img_pl: imgs}
        pred_ = session.run(pred, feed_dict=feed_dict_flip)
        for i in range(len(pred_)):
            if pred_[i] == 1:
                flipped_index.append(index[i])
        if len(flipped_index) != 0:
            return True
        else:
            return False

    @staticmethod
    def flipping_words(imgs):
        imgs_list = []
        num_imgs = len(imgs)
        for i in range(num_imgs):
            img = imgs[num_imgs - i - 1]
            flipped_img = np.rot90(img, 2)
            imgs_list.append(flipped_img)
        flipped_imgs = np.reshape(imgs_list, (-1, 28, 28, 1))
        return flipped_imgs
