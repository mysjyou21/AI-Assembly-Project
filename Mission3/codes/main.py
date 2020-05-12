import tensorflow as tf
import glob
import os
from DCA_stefan import DCA
from args import define_args

FLAGS = define_args()

    
def main(args):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    
    model = DCA(FLAGS)

    if 'train' in FLAGS.mode:
        model.train()
    elif FLAGS.mode == 'test':
        model.test()

if __name__=='__main__':
    tf.app.run()
