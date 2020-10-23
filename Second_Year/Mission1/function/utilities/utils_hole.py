import json

# pre-defined hole information
part_holes = {}
part_holes['stefan_part1'] = []
part_holes['stefan_part2'] = ['00#00#122620', '00#01#122620', '01#00#', '01#01#', '01#02#', '03#00#', '03#01#', '03#02#']
#part_holes['stefan_part2'] = ['00#00#122620', '00#01#122620', '01#00#101350', '01#01#104322', '01#02#101350', '03#00#101350', '03#01#104322', '03#02#101350']
part_holes['stefan_part3'] = ['00#00#122620', '00#01#122620', '01#00#', '01#01#', '01#02#', '03#00#', '03#01#', '03#02#']
#part_holes['stefan_part3'] = ['00#00#122620', '00#01#122620', '01#00#101350', '01#01#104322', '01#02#101350', '03#00#101350', '03#01#104322', '03#02#101350']
part_holes['stefan_part4'] = ['01#00#101350', '01#01#', '01#02#101350', '01#03#101350', '03#00#101350', '03#01#', '03#02#101350', '03#03#101350']
#part_holes['stefan_part4'] = ['01#00#101350', '01#01#104322', '01#02#101350', '01#03#101350', '03#00#101350', '03#01#104322', '03#02#101350', '03#03#101350']
part_holes['stefan_part5'] = ['00#00#', '00#01#', '00#02#', '00#03#', '00#04#', '00#05#', '00#06#', '00#07#', '00#08#','00#09#', '02#00#104322', '02#01#104322', '02#02#104322']
 # '02#00' == '00#01', '02#01' == '00#06', '02#02' == '00#07'
#part_holes['stefan_part5'] = ['00#00#', '00#01#104322', '00#02#', '00#03#', '00#04#', '00#05#', '00#06#104322', '00#07#104322', '00#08#','00#09#', '02#00#104322', '02#01#104322', '02#02#104322'] # '02#00' == '00#01', '02#01' == '00#06', '02#02' == '00#07'
part_holes['stefan_part6'] = ['00#00#', '00#01#', '00#02#', '02#00#', '02#01#', '02#02#', '02#03#', '02#04#', '02#05#101350', '02#06#', '02#07#', '02#08#', '02#09#101350']
 # '00#00'=='02#01', '00#01' == '02#06', '00#02' == '02#07'
#part_holes['stefan_part6'] = ['00#00#104322', '00#01#104322', '00#02#104322', '02#00#101350', '02#01#104322', '02#02#101350', '02#03#101350', '02#04#101350', '02#05#101350', '02#06#104322', '02#07#104322', '02#08#101350', '02#09#101350'] # '00#00'=='02#01', '00#01' == '02#06', '00#02' == '02#07'

part_holes['stefan12_step1_a'] = ['01#00#101350', '01#01#', '01#02#101350', '03#00#101350', '03#01#', '03#02#101350']
 # 'stefan12_step1_a'-'stefan12_step2': '03#00'-'00#07', '03#02'-'00#05'
part_holes['stefan12_step1_b'] = ['01#00#101350', '01#01#', '01#02#101350', '03#00#101350', '03#01#', '03#02#101350']
 # 'stefan12_step1_b'-'stefan_part6': '01#00'-'02#05', '01#02'-'02#09'
part_holes['stefan12_step2'] = ['04#00#', '04#01#', '04#02#', '05#00#101350', '05#01#', '05#02#101350', '05#03#', '05#04#', '05#05#', '05#06#', '05#07#']
 # '00#00'=='02#00', '00#04'=='02#01', '00#06'=='02#02'
part_holes['stefan12_step3'] = ['04#00#', '04#01#', '04#02#', '05#00', '05#01#', '05#02#101350', '05#03#101350', '05#04#', '05#05#101350']
 # '00#01'=='02#00', '00#04'=='02#01', '00#05'=='02#02'; 'stefan12_step3'-'stefan_part4': '00#00'-'01#00', '00#02'-'01#02', '00#03'-'01#03'
part_holes['stefan12_step4'] = ['04#00#', '04#01#', '04#02#', '05#00#104322', '05#01#104322', '05#02#104322']
 # 'stefan12_step4'-'stefan_part5': '00#00'-'02#00', '00#01'-'02#01', '00#02'-'02#02'
part_holes['stefan12_step5'] = ['04#00#104322', '04#01#104322', '04#02#104322', '05#00#', '05#01#', '05#02#']
part_holes['stefan12_step6'] = []
part_holes['stefan12_step7'] = []
part_holes['stefan12_step8'] = ['00#00#122925', '00#01#122925', '00#02#122925', '00#03#122925']
part_holes['stefan12_step9'] = []

part_holes['stefan12_step2#stefan12_step1_a'] = ['05#02#01#00', '05#00#01#02']
part_holes['stefan12_step1_b#stefan_part6'] = ['03#02#02#05', '03#02#02#09']
part_holes['stefan_part4#stefan12_step3'] = ['01#00#05#05', '01#02#05#03', '01#03#05#02']
part_holes['stefan_part5#stefan12_step4'] = ['02#00#05#02', '02#01#05#01', '02#02#05#00']

#with open('./hole.json', 'w') as f:
#   json.dump(part_holes, f, indent=2)

part_holes = {}
part_holes['step_2#step_1_a'] = ['05#02#01#00', '05#00#01#02']
part_holes['step_1_b#part6'] = ['03#02#bottom#5', '03#02#bottom#3']
part_holes['part4#step_3'] = ['right#7#05#05', 'right#3#05#03', 'right#1#05#02']
part_holes['part5#step_4'] = ['bottom#8#05#02', 'bottom#1#05#01', 'bottom#4#05#00']


# pre-defined pose - related information
pose_dic = {} # key: pose_id, value: [view_id(-up), view_id(-down)]
pose_dic[0] = pose_dic[1] = ['left', 'right'] #0,0,0
pose_dic[2] = pose_dic[3] = ['front', 'back'] #0,0,90
pose_dic[4] = pose_dic[5] = ['right', 'left'] #0,0,180
pose_dic[6] = pose_dic[7] = ['back', 'front'] #0,0,270
pose_dic[8] = pose_dic[9] = ['top', 'bottom'] #90,0,0
pose_dic[10] = pose_dic[11] = ['front', 'back'] #90,0,90
pose_dic[12] = pose_dic[13] = ['bottom', 'top'] #90,0,180
pose_dic[14] = pose_dic[15] = ['back', 'front'] #90,0,270
pose_dic[16] = pose_dic[17] = ['top', 'bottom'] #90,90,0
pose_dic[18] = pose_dic[19] = ['right', 'left'] #90,90,90
pose_dic[20] = pose_dic[21] = ['bottom', 'top'] #90,90,180
pose_dic[22] = pose_dic[23] = ['left', 'right'] #90,90,270
pose_dic[24] = pose_dic[25] = ['top', 'bottom'] #90,180,0
pose_dic[26] = pose_dic[27] = ['back', 'front'] #90,180,90
pose_dic[28] = pose_dic[29] = ['bottom', 'top'] #90,180,180
pose_dic[30] = pose_dic[31] = ['front', 'back'] #90,180,270
pose_dic[32] = pose_dic[33] = ['top', 'bottom'] #90,270,0
pose_dic[34] = pose_dic[35] = ['left', 'right'] #90,270,90
pose_dic[36] = pose_dic[37] = ['bottom', 'top'] #90,270,180
pose_dic[38] = pose_dic[39] = ['right', 'left'] #90,270,270
pose_dic[40] = pose_dic[41] = ['right', 'left'] #180,0,0
pose_dic[42] = pose_dic[43] = ['front', 'back'] #180,0,90
pose_dic[44] = pose_dic[45] = ['left', 'right'] #180,0,180
pose_dic[46] = pose_dic[47] = ['back', 'front'] #180,0,270

#with open('./hole_pose.json', 'w') as f:
#   json.dump(pose_dic, f, indent=2)


# pre-defined pose - related information
pose_dic = {} # key: pose_id, value: [view_id(-up), view_id(-down)]
pose_dic[0] = pose_dic[1] = ['left', 'right'] #0,0,0
pose_dic[2] = pose_dic[3] = ['front', 'back'] #0,0,90
pose_dic[4] = pose_dic[5] = ['right', 'left'] #0,0,180
pose_dic[6] = pose_dic[7] = ['back', 'front'] #0,0,270
pose_dic[8] = pose_dic[9] = ['top', 'bottom'] #90,0,0
pose_dic[10] = pose_dic[11] = ['front', 'back'] #90,0,90
pose_dic[12] = pose_dic[13] = ['bottom', 'top'] #90,0,180
pose_dic[14] = pose_dic[15] = ['back', 'front'] #90,0,270
pose_dic[16] = pose_dic[17] = ['top', 'bottom'] #90,90,0
pose_dic[18] = pose_dic[19] = ['right', 'left'] #90,90,90
pose_dic[20] = pose_dic[21] = ['bottom', 'top'] #90,90,180
pose_dic[22] = pose_dic[23] = ['left', 'right'] #90,90,270
pose_dic[24] = pose_dic[25] = ['top', 'bottom'] #90,180,0
pose_dic[26] = pose_dic[27] = ['back', 'front'] #90,180,90
pose_dic[28] = pose_dic[29] = ['bottom', 'top'] #90,180,180
pose_dic[30] = pose_dic[31] = ['front', 'back'] #90,180,270
pose_dic[32] = pose_dic[33] = ['top', 'bottom'] #90,270,0
pose_dic[34] = pose_dic[35] = ['left', 'right'] #90,270,90
pose_dic[36] = pose_dic[37] = ['bottom', 'top'] #90,270,180
pose_dic[38] = pose_dic[39] = ['right', 'left'] #90,270,270
pose_dic[40] = pose_dic[41] = ['right', 'left'] #180,0,0
pose_dic[42] = pose_dic[43] = ['front', 'back'] #180,0,90
pose_dic[44] = pose_dic[45] = ['left', 'right'] #180,0,180
pose_dic[46] = pose_dic[47] = ['back', 'front'] #180,0,270

#with open('./hole_pose_part.json', 'w') as f:
#   json.dump(pose_dic, f, indent=2)

loc_dic = {}
# 'B', 'F', 'R', 'L' ['up-up', 'up-left', 'down-up', 'down-left']
loc_dic[0] = loc_dic[1] = ['B','R','B','L'] #0,0,0
loc_dic[2] = loc_dic[3] = ['B','R','B','L'] #0,0,90
loc_dic[4] = loc_dic[5] = ['B','R','B','L'] #0,0,180
loc_dic[6] = loc_dic[7] = ['B','R','B','L'] #0,0,270
loc_dic[8] = loc_dic[9] = ['R','F','L','F'] #90,0,0
loc_dic[10] = loc_dic[11] = ['R','F','R','B'] #90,0,90
loc_dic[12] = loc_dic[13] = ['R','F','L','F'] #90,0,180
loc_dic[14] = loc_dic[15] = ['L','B','L','F'] #90,0,270
loc_dic[16] = loc_dic[17] = ['B','R','F','R'] #90,90,0
loc_dic[18] = loc_dic[19] = ['R','F','R','B'] #90,90,90
loc_dic[20] = loc_dic[21] = ['F','L','B','L'] #90,90,180
loc_dic[22] = loc_dic[23] = ['L','B','L','F'] #90,90,270
loc_dic[24] = loc_dic[25] = ['L','B','R','B'] #90,180,0
loc_dic[26] = loc_dic[27] = ['R','F','R','B'] #90,180,90
loc_dic[28] = loc_dic[29] = ['L','B','R','B'] #90,180,180
loc_dic[30] = loc_dic[31] = ['L','B','L','F'] #90,180,270
loc_dic[32] = loc_dic[33] = ['F','L','B','L'] #90,270,0
loc_dic[34] = loc_dic[35] = ['R','F','R','B'] #90,270,90
loc_dic[36] = loc_dic[37] = ['B','R','F','R'] #90,270,180
loc_dic[38] = loc_dic[39] = ['L','B','L','F'] #90,270,270
loc_dic[40] = loc_dic[41] = ['F','L','F','R'] #180,0,0
loc_dic[42] = loc_dic[43] = ['F','L','F','R'] #180,0,90
loc_dic[44] = loc_dic[45] = ['F','L','F','R'] #180,0,180
loc_dic[46] = loc_dic[47] = ['F','L','F','R'] #180,0,270
#with open('./hole_loc_part.json', 'w') as f:
#   json.dump(loc_dic, f, indent=2)

# loc_dic = {}
# # 'B', 'F', 'R', 'L' ['up-up', 'up-left', 'down-up', 'down-left']
# loc_dic[0] = loc_dic[1] = ['R','B','R','F'] #0,0,0
# loc_dic[2] = loc_dic[3] = ['F','R','B','R'] #0,0,90
# loc_dic[4] = loc_dic[5] = ['L','F','L','B'] #0,0,180
# loc_dic[6] = loc_dic[7] = ['B','L','F','L'] #0,0,270
# loc_dic[8] = loc_dic[9] = ['R','B','R','F'] #90,0,0
# loc_dic[10] = loc_dic[11] = ['R','B','L','B'] #90,0,90
# loc_dic[12] = loc_dic[13] = ['L','F','L','B'] #90,0,180
# loc_dic[14] = loc_dic[15] = ['R','B','L','B'] #90,0,270
# loc_dic[16] = loc_dic[17] = ['F','R','F','L'] #90,90,0
# loc_dic[18] = loc_dic[19] = ['F','R','F','L'] #90,90,90
# loc_dic[20] = loc_dic[21] = ['F','R','F','L'] #90,90,180
# loc_dic[22] = loc_dic[23] = ['F','R','F','L'] #90,90,270
# loc_dic[24] = loc_dic[25] = ['R','B','R','F'] #90,180,0
# loc_dic[26] = loc_dic[27] = ['L','F','R','F'] #90,180,90
# loc_dic[28] = loc_dic[29] = ['R','B','R','F'] #90,180,180
# loc_dic[30] = loc_dic[31] = ['L','F','R','F'] #90,180,270
# loc_dic[32] = loc_dic[33] = ['B','L','B','R'] #90,270,0
# loc_dic[34] = loc_dic[35] = ['B','L','B','R'] #90,270,90
# loc_dic[36] = loc_dic[37] = ['B','L','B','R'] #90,270,180
# loc_dic[38] = loc_dic[39] = ['B','L','B','R'] #90,270,270
# loc_dic[40] = loc_dic[41] = ['R','B','R','F'] #180,0,0
# loc_dic[42] = loc_dic[43] = ['B','L','F','L'] #180,0,90
# loc_dic[44] = loc_dic[45] = ['L','F','L','B'] #180,0,180
# loc_dic[46] = loc_dic[47] = ['F','R','B','R'] #180,0,270
#with open('./function/utilities/hole_loc.json', 'w') as f:
#    json.dump(loc_dic, f, indent=2)

loc_dic = {}
# 'B', 'F', 'R', 'L' ['up-up', 'up-left', 'down-up', 'down-left']
loc_dic[0] = loc_dic[1] = ['B','R','B','L'] #0,0,0
loc_dic[2] = loc_dic[3] = ['B','R','B','L'] #0,0,90
loc_dic[4] = loc_dic[5] = ['B','R','B','L'] #0,0,180
loc_dic[6] = loc_dic[7] = ['B','R','B','L'] #0,0,270
loc_dic[8] = loc_dic[9] = ['R','F','L','F'] #90,0,0
loc_dic[10] = loc_dic[11] = ['R','F','R','B'] #90,0,90
loc_dic[12] = loc_dic[13] = ['R','F','L','F'] #90,0,180
loc_dic[14] = loc_dic[15] = ['L','B','L','F'] #90,0,270
loc_dic[16] = loc_dic[17] = ['B','R','F','R'] #90,90,0
loc_dic[18] = loc_dic[19] = ['R','F','R','B'] #90,90,90
loc_dic[20] = loc_dic[21] = ['F','L','B','L'] #90,90,180
loc_dic[22] = loc_dic[23] = ['L','B','L','F'] #90,90,270
loc_dic[24] = loc_dic[25] = ['L','B','R','B'] #90,180,0
loc_dic[26] = loc_dic[27] = ['R','F','R','B'] #90,180,90
loc_dic[28] = loc_dic[29] = ['L','B','R','B'] #90,180,180
loc_dic[30] = loc_dic[31] = ['L','B','L','F'] #90,180,270
loc_dic[32] = loc_dic[33] = ['F','L','B','L'] #90,270,0
loc_dic[34] = loc_dic[35] = ['R','F','R','B'] #90,270,90
loc_dic[36] = loc_dic[37] = ['B','R','F','R'] #90,270,180
loc_dic[38] = loc_dic[39] = ['L','B','L','F'] #90,270,270
loc_dic[40] = loc_dic[41] = ['F','L','F','R'] #180,0,0
loc_dic[42] = loc_dic[43] = ['F','L','F','R'] #180,0,90
loc_dic[44] = loc_dic[45] = ['F','L','F','R'] #180,0,180
loc_dic[46] = loc_dic[47] = ['F','L','F','R'] #180,0,270
#with open('./hole_loc.json', 'w') as f:
#   json.dump(loc_dic, f, indent=2)

import sys
#sys.path.append('../function')
#sys.path.append('../function/Pose')
#from pose_gt import *

#LABEL_TO_POSE = {}
#for key in POSE_TO_LABEL.keys():
#    value = POSE_TO_LABEL[key]
#    LABEL_TO_POSE[value] = key

#with open('./label_to_pose.json', 'w') as f:
#    json.dump(LABEL_TO_POSE, f, indent=2)

# STEFAN INTERMEDIATES(step) HOLE LABELING - faces
stefan_inter = {}
#stefan_inter['step1_a'] = {'part2_1-hole_1': ['left', '101350'], 'part2_1-hole_2': ['right', '101350'], 'part2_1-hole_3': ['left', '104322'], 'part2_1-hole_4': ['right', '104322'], 'part2_1-hole_5': ['left', '101350'], 'part2_1-hole_6': ['right', '101350'], 'C122620_1-hole_1': ['back', '122620'], 'C122620_2-hole_1': ['back', '122620']}
#stefan_inter['step1_b'] = {'part3_1-hole_1': ['left', '101350'], 'part3_1-hole_2': ['right', '101350'], 'part3_1-hole_3': ['left', '104322'], 'part3_1-hole_4': ['right', '104322'], 'part3_1-hole_5': ['left', '101350'], 'part3_1-hole_6': ['right', '101350'], 'C122620_1-hole_1': ['back', '122620'], 'C122620_2-hole_1': ['back', '122620']}
stefan_inter['step1_a'] = {'part2_1-hole_1': 'left', 'part2_1-hole_2': 'right', 'part2_1-hole_3': 'left', 'part2_1-hole_4': 'right', 'part2_1-hole_5': 'left', 'part2_1-hole_6': 'right', 'C122620_1-hole_1': 'back', 'C122620_2-hole_1': 'back'}
stefan_inter['step1_b'] = {'part3_1-hole_1': 'left', 'part3_1-hole_2': 'right', 'part3_1-hole_3': 'left', 'part3_1-hole_4': 'right', 'part3_1-hole_5': 'left', 'part3_1-hole_6': 'right', 'C122620_1-hole_1': 'back', 'C122620_2-hole_1': 'back'}
stefan_inter['step2'] = {'part3_1-hole_4': 'bottom', 'part6_1-hole_1': 'bottom', 'part6_1-hole_2': 'top', 'part6_1-hole_3': 'bottom', 'part6_1-hole_5': 'top', 'part6_1-hole_7': 'bottom', 'part6_1-hole_8': 'bottom', 'part6_1-hole_9': 'top', 'part6_1-hole_10': 'bottom', 'C122620_3_hole_1': 'front', 'C122620_4_hole_1': 'front'}
stefan_inter['step3'] = {'part2_1-hole_3': 'bottom', 'part3_1-hole_4': 'bottom', 'part6_1-hole_2': 'top', 'part6_1-hole_5': 'top', 'part6_1-hole_7': 'bottom', 'part6_1-hole_8': 'bottom', 'part6_1-hole_9': 'top', 'part6_1-hole_10': 'bottom', 'C122620_1_hole_1': 'front', 'C122620_2_hole_1': 'front', 'C122620_3_hole_1': 'front', 'C122620_4_hole_1': 'front'}
stefan_inter['step4'] = {'part2_1-hole_3': 'bottom', 'part3_1-hole_4': 'bottom', 'part4_1-hole_5': 'bottom', 'part6_1-hole_2': 'top', 'part6_1-hole_5': 'top', 'part6_1-hole_7': 'bottom', 'part6_1-hole_8': 'bottom', 'part6_1-hole_9': 'top', 'part6_1-hole_10': 'bottom', 'C122620_1_hole_1': 'front', 'C122620_2_hole_1': 'front', 'C122620_3_hole_1': 'front', 'C122620_4_hole_1': 'front'}
stefan_inter['step5'] = {'part6_1-hole_2': 'top', 'part6_1-hole_5': 'top', 'part6_1-hole_9': 'top', 'C122620_1_hole_1': 'front', 'C122620_2_hole_1': 'front', 'C122620_3_hole_1': 'front', 'C122620_4_hole_1': 'front'}
stefan_inter['step6'] = {'C122620_1_hole_1': 'front', 'C122620_2_hole_1': 'front', 'C122620_3_hole_1': 'front', 'C122620_4_hole_1': 'front'}
stefan_inter['step7'] = {'C122620_1_hole_1': 'front', 'C122620_2_hole_1': 'front', 'C122620_3_hole_1': 'front', 'C122620_4_hole_1': 'front'}
stefan_inter['step8'] = {'C122620_1_hole_1': 'front', 'C122620_2_hole_1': 'front', 'C122620_3_hole_1': 'front', 'C122620_4_hole_1': 'front'}

# pre-defined hole-connectivity 'PART1(related to fasteners)#PART2(may not having fasteners)'
connect = {}
connect['step1_a#step2'] = ['left#part2_1-hole_2#bottom#part6_1-hole_3', 'left#part2_1-hole_6#bottom#part6_1-hole_1']
connect['step1_b#part6'] = ['right#part3_1-hole_1#bottom#5', 'right#part3_1-hole_5#bottom#3']
connect['part4#step3'] = ['right#7#bottom#part6_1-hole_10', 'right#3#bottom#part6_1-hole_8', 'right#1#bottom#part6_1-hole_7']
connect['part5#step4'] = ['bottom#8#bottom#part4_1-hole_5', 'bottom#1#bottom#part2_1-hole_3', 'bottom#4#bottom#part3_1-hole_4']

connector = {}
connector[3.] = '122620'
connector[4.] = '101350'
connector[3.5] = '101350'
connector[2.5] = '104322'
connector[6.3] = '104322'
connector[2.75] = '104322'

############ READ PART CSV FILE
import csv
import json
import pandas as pd
import glob, os
import numpy as np

sort_key= {}
#sort_key['top'] = ['x', False, 'y', True]
#sort_key['left'] = ['z', True, 'y', True]
#sort_key['right'] = ['z', False, 'y', True]
#sort_key['bottom'] = ['x', True, 'y', True]
#sort_key['front'] = ['x', False, 'z', True]
#sort_key['back'] = ['x', False, 'z', False]
sort_key['top'] = ['x', False, 'y', True]
sort_key['left'] = ['y', True, 'z', True] #'z', False, 'y', True]
sort_key['right'] = ['y', False, 'z', True] #'z', True, 'y', True]
sort_key['bottom'] = ['x', False, 'y', False]
sort_key['front'] = ['x', False, 'z', True]
sort_key['back'] = ['x', True, 'z', True]


def read_csv(csv_dir):
    hole_dic = {}
    csv_paths = glob.glob(os.path.join(csv_dir, '*.csv'))
    part_paths = []
    step_paths = []
    connector_paths = []
    for path in csv_paths:
        if 'part' in os.path.basename(path):
            part_paths.append(path)
        elif 'step' in os.path.basename(path):
            step_paths.append(path)
        else:
            connector_paths.append(path)

    for path in part_paths:
        # read csv file
        part_name = os.path.basename(path).rstrip('.csv')
        data = pd.read_csv(path, header=None)
        if data.shape[0] == 2: #no holes
            hole_dic[part_name] = []
            continue
        holes_info = data.loc[3:][[0,1,2,3,7,9,11]] # hole_num, centerx, centery, centerz, radius, through, direction
        part_holes = {}
        for i in range(3, holes_info.shape[0]+3):
            hole_info = holes_info.loc[i,:]
            name = hole_info[0]
            x,y,z = float(hole_info[1]),float(hole_info[2]),float(hole_info[3])
            radius = float(hole_info[7])
            con = connector[radius]
            through = json.loads(hole_info[9].lower())
            direction = hole_info[11]
            part_holes[direction] = part_holes.get(direction, []) + [[name, x,y,z, con, through, direction]]

        # sort
        for k,v in part_holes.items():
            temp = v.copy()
            left_key = 1 if sort_key[k][0] == 'x' else 2 if sort_key[k][0] == 'y' else 3
            left_order = sort_key[k][1]
            up_key = 1 if sort_key[k][2] == 'x' else 2 if sort_key[k][2] == 'y' else 3
            up_order = sort_key[k][3]
            temp = sorted(sorted(temp, key=lambda x: x[left_key], reverse=left_order), key=lambda x: x[up_key], reverse=up_order)
            part_holes[k] = temp

        for k,v in part_holes.items():
            hole_dic[part_name] = hole_dic.get(part_name, []) + v

    for path in step_paths:
        part_name = os.path.basename(path).rstrip('.csv')
        data = pd.read_csv(path, header=None)
        if data.shape[0] == 2: #no holes
            hole_dic[part_name] = []
            continue
        holes_info = data.loc[3:][:] # hole_num, centerx, centery, centerz
        part_holes = {}
        is_null = pd.isnull(holes_info)
        for i in range(3, holes_info.shape[0]+3):
            if is_null.loc[i,0]:
                continue
            hole_info = holes_info.loc[i,:]
            name = hole_info[0]
            if 'C122620' in name:
                con = '122925'
                through = False
            else:
                original_name = name.split('-')[0].split('_')[0]
                original_hole_name = str(int(name.split('-')[1].split('_')[1])-1)
                original_hole_info = hole_dic[original_name]
                original_hole_info = [x for x in original_hole_info if x[0] == original_hole_name][0]
                con = original_hole_info[4]
                through = original_hole_info[5]
            x,y,z = float(hole_info[1]), float(hole_info[2]), float(hole_info[3])
            direction = stefan_inter[part_name.rstrip('.stl')][name]
            part_holes[direction] = part_holes.get(direction, []) + [[name, x,y,z, con, through, direction]]

        for k,v in part_holes.items():
            temp = v.copy()
            left_key = 1 if sort_key[k][0] == 'x' else 2 if sort_key[k][0] == 'y' else 3
            left_order = sort_key[k][1]
            up_key = 1 if sort_key[k][2] == 'x' else 2 if sort_key[k][2] == 'y' else 3
            up_order = sort_key[k][3]
            temp = sorted(sorted(temp, key=lambda x: x[left_key], reverse=left_order), key=lambda x: x[up_key], reverse=up_order)
            part_holes[k] = temp

        for k,v in part_holes.items():
            hole_dic[part_name] = hole_dic.get(part_name, []) + v

    for k,v in connect.items():
        hole_dic[k] = v

    return hole_dic
