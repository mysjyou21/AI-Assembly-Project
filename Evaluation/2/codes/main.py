import tensorflow as tf
import glob
import os
from DCA import DCA
from args import define_args

FLAGS = define_args()

    
def main(args):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    
    model = DCA(FLAGS)

    model.test()

if __name__=='__main__':
    tf.app.run()
