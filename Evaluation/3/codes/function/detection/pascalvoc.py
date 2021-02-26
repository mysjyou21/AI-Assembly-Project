###########################################################################################
#                                                                                         #
# This sample shows how to evaluate object detections applying the following metrics:     #
#  * Precision x Recall curve       ---->       used by VOC PASCAL 2012)                  #
#  * Average Precision (AP)         ---->       used by VOC PASCAL 2012)                  #
#                                                                                         #
# Developed by: Rafael Padilla (rafael.padilla@smt.ufrj.br)                               #
#        SMT - Signal Multimedia and Telecommunications Lab                               #
#        COPPE - Universidade Federal do Rio de Janeiro                                   #
#        Last modification: Oct 9th 2018                                                 #
###########################################################################################

import argparse
import glob
import os
import shutil
# from argparse import RawTextHelpFormatter
import sys

from lib.BoundingBox import BoundingBox
from lib.BoundingBoxes import BoundingBoxes
from lib.Evaluator import *
from lib.utils import BBFormat

from function.utils import *

class pascalvoc():

    def __init__(self, args):
        # Load detection model.
        self.args = args
        self.errors = []
        self.imgSize = (0, 0)

        self.gtfolder = os.path.join('../data/detection/bbox_answer')
        self.detfolder = os.path.join('./intermediate_results/detection')

        # Parameters
        self.iouThreshold=0.5

    def getBoundingBoxes(self,
                         directory,
                         isGT,
                         bbFormat,
                         coordType,
                         allBoundingBoxes=None,
                         allClasses=None,
                         imgSize=(0, 0)):
        """Read txt files containing bounding boxes (ground truth and detections)."""
        if allBoundingBoxes is None:
            allBoundingBoxes = BoundingBoxes()
        if allClasses is None:
            allClasses = []
        # Read ground truths
        cwd = os.getcwd()

        # os.chdir(os.path.join(directory))
        os.chdir(directory)
        files = glob.glob("*.txt")
        files.sort()

        for f in files:
            nameOfImage = f.replace(".txt", "")
            fh1 = open(f, "r")
            for line in fh1:
                line = line.replace("\n", "")
                if line.replace(' ', '') == '':
                    continue
                splitLine = line.split(" ")
                if isGT:
                    # idClass = int(splitLine[0]) #class
                    idClassList = (splitLine[:-4])  # class
                    idClass = ''
                    for n in range(len(idClassList)):
                        idClass += idClassList[n]
                    x = float(splitLine[-4])
                    y = float(splitLine[-3])
                    w = float(splitLine[-2])
                    h = float(splitLine[-1])
                    bb = BoundingBox(
                        nameOfImage,
                        idClass,
                        x,
                        y,
                        w,
                        h,
                        coordType,
                        imgSize,
                        BBType.GroundTruth,
                        format=bbFormat)
                else:
                    # idClass = int(splitLine[0]) #class
                    idClassList = (splitLine[:-5])  # class
                    idClass = ''
                    for n in range(len(idClassList)):
                        idClass += idClassList[n]
                    confidence = float(splitLine[-5])
                    x = float(splitLine[-4])
                    y = float(splitLine[-3])
                    w = float(splitLine[-2])
                    h = float(splitLine[-1])
                    bb = BoundingBox(
                        nameOfImage,
                        idClass,
                        x,
                        y,
                        w,
                        h,
                        coordType,
                        imgSize,
                        BBType.Detected,
                        confidence,
                        format=bbFormat)
                allBoundingBoxes.addBoundingBox(bb)
                if idClass not in allClasses:
                    allClasses.append(idClass)
            fh1.close()
        os.chdir(cwd)
        return allBoundingBoxes, allClasses


    def pascalvoc_calculate_iou(self):
        # Arguments validation

        # Validate formats


        # Get groundtruth boxes
        allBoundingBoxes, allClasses = self.getBoundingBoxes(
            self.gtfolder, True, BBFormat.XYX2Y2, CoordinatesType.Absolute, imgSize=self.imgSize)
        # Get detected boxes
        allBoundingBoxes, allClasses = self.getBoundingBoxes(
            self.detfolder, False, BBFormat.XYX2Y2, CoordinatesType.Absolute, allBoundingBoxes, allClasses, imgSize=self.imgSize)
        allClasses.sort()

        evaluator = Evaluator()
        self.acc_AP = 0
        self.validClasses = 0

        # Plot Precision x Recall curve
        detections = evaluator.PlotPrecisionRecallCurve(
            allBoundingBoxes,  # Object containing all bounding boxes (ground truths and detections)
            IOUThreshold=self.iouThreshold,  # IOU threshold
            method=MethodAveragePrecision.EveryPointInterpolation,
            showAP=True,  # Show Average Precision in the title of the plot
            showInterpolatedPrecision=False,  # Don't plot the interpolated precision curve
            savePath=None,
            showGraphic=False)


        print('---------------')
        # each detection is a class
        for metricsPerClass in detections:

            # Get metric values per each class
            cl = metricsPerClass['class']
            ap = metricsPerClass['AP']
            precision = metricsPerClass['precision']
            recall = metricsPerClass['recall']
            totalPositives = metricsPerClass['total positives']
            total_TP = metricsPerClass['total TP']
            total_FP = metricsPerClass['total FP']

            if totalPositives > 0:
                self.validClasses = self.validClasses + 1
                self.acc_AP = self.acc_AP + ap
                prec = ['%.2f' % p for p in precision]
                rec = ['%.2f' % r for r in recall]
                ap_str = "{0:.2f}%".format(ap * 100)
                # ap_str = "{0:.4f}%".format(ap * 100)
                print('AP: %s (%s)' % (ap_str, cl))

        mAP = self.acc_AP / self.validClasses
        mAP_str = "{0:.2f}%".format(mAP * 100)
        print('mAP: %s' % mAP_str)
        # f.write('\n\n\nmAP: %s' % mAP_str)
        return mAP * 100
