import glob
import os
import shutil
import sys

from lib.BoundingBox import BoundingBox
from lib.BoundingBoxes import BoundingBoxes
from lib.Evaluator import *
from lib.utils import BBFormat


def getBoundingBoxes(directory,
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
    os.chdir(directory)
    files = glob.glob("*.txt")
    files.sort()
    # Read GT detections from txt file
    # Each line of the files in the groundtruths folder represents a ground truth bounding box
    # (bounding boxes that a detector should detect)
    # Each value of each line is  "class_id, x, y, width, height" respectively
    # Class_id represents the class of the bounding box
    # x, y represents the most top-left coordinates of the bounding box
    # x2, y2 represents the most bottom-right coordinates of the bounding box
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


# Get current path to set default folders



def pascalvoc_calculate_iou(gtfolder, detfolder):
    currentPath = os.path.dirname(os.path.abspath(__file__))
    iouThreshold = 0.5
    # Validate formats
    gtFormat = 2    # xyrb
    detFormat = 2   # xyrb
    # Groundtruth folder
    gtFolder = './input/test/det_anns_indiv'
    # Coordinates types
    gtCoordType = 2     # absolute
    detCoordType = 2    # absolute
    imgSize = (0, 0)
    # Detection folder
    detFolder = os.path.join(currentPath, 'detections')
    # folder to save result plots
    savePath = os.path.join(currentPath, 'results')
    shutil.rmtree(savePath, ignore_errors=True)  # Clear folder
    os.makedirs(savePath)
    
    # print('iouThreshold= %f' % iouThreshold)
    # print('savePath = %s' % savePath)
    # print('gtFormat = %s' % gtFormat)
    # print('detFormat = %s' % detFormat)
    # print('gtFolder = %s' % gtFolder)
    # print('detFolder = %s' % detFolder)
    # print('gtCoordType = %s' % gtCoordType)
    # print('detCoordType = %s' % detCoordType)

    # Get groundtruth boxes
    allBoundingBoxes, allClasses = getBoundingBoxes(
        gtFolder, True, gtFormat, gtCoordType, imgSize=imgSize)
    # Get detected boxes
    allBoundingBoxes, allClasses = getBoundingBoxes(
        detFolder, False, detFormat, detCoordType, allBoundingBoxes, allClasses, imgSize=imgSize)
    allClasses.sort()

    evaluator = Evaluator()
    acc_AP = 0
    validClasses = 0

    # Plot Precision x Recall curve
    detections = evaluator.PlotPrecisionRecallCurve(
        allBoundingBoxes,  # Object containing all bounding boxes (ground truths and detections)
        IOUThreshold=iouThreshold,  # IOU threshold
        method=MethodAveragePrecision.EveryPointInterpolation,
        showAP=True,  # Show Average Precision in the title of the plot
        showInterpolatedPrecision=False,  # Don't plot the interpolated precision curve
        savePath=savePath)

    f = open(os.path.join(savePath, 'results.txt'), 'w')
    f.write('Object Detection Metrics\n')
    f.write('Average Precision (AP), Precision and Recall per class:')
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
            validClasses = validClasses + 1
            acc_AP = acc_AP + ap
            prec = ['%.2f' % p for p in precision]
            rec = ['%.2f' % r for r in recall]
            ap_str = "{0:.2f}%".format(ap * 100)
            # ap_str = "{0:.4f}%".format(ap * 100)
            print('AP: %s (%s)' % (ap_str, cl))
            f.write('\n\nClass: %s' % cl)
            f.write('\nAP: %s' % ap_str)
            f.write('\nPrecision: %s' % prec)
            f.write('\nRecall: %s' % rec)

    mAP = acc_AP / validClasses
    mAP_str = "{0:.2f}%".format(mAP * 100)
    print('mAP: %s' % mAP_str)
    f.write('\n\n\nmAP: %s' % mAP_str)
    return mAP * 100
