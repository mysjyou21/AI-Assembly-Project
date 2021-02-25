import time
start_time = time.time()
import glob
import os
import cv2
import keras
from args import define_args
from function.utils import print_time
from quantitative_report_pose import QuantReportPose

args = define_args()

def main():
    # set gpu
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print('Using gpu : {}\n'.format(args.gpu))

    # load models
    if not args.mode == 'test_data':
        print('----------------')
        print('Loading Weights')
        print('----------------')
    POSE = QuantReportPose(args)

    # execution
    if args.mode in ['detection', 'detection_unit_test']:
        POSE.detection()
    if args.mode in ['retrieval_unit_test']:
        POSE.retrieval()
    if args.mode in ['pose_unit_test']:
        POSE.pose()
        if args.output_visualization: # temp blue
            POSE.output_visualization() # temp blue
    if args.mode in ['retrieval']:
        POSE.detection()
        POSE.retrieval()
    if args.mode in ['pose']:
        POSE.detection()
        POSE.retrieval()
        POSE.pose()
    if args.mode in ['test']:
        POSE.detection()
        POSE.retrieval()
        POSE.pose()
        if args.output_visualization:
            POSE.output_visualization()
    if args.mode in ['test_data']:
        POSE.save_test_data()

if __name__ == '__main__':
    main()
    keras.backend.clear_session() 
    end_time = time.time()
    print('execution time : {:.3}'.format(end_time - start_time))

