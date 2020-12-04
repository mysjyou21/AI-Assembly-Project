import json

pose_key=False
if pose_key:
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

loc_key=False
if loc_key:
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

# STEFAN BASE PARTS(part) HOLE LABELING - faces
stefan_part = {}
stefan_part['part1'] = {'hole_1': 'top', 'hole_2': 'top', 'hole_3': 'top', 'hole_4': 'top'}
stefan_part['part2'] = {'hole_1': 'left', 'hole_2': 'right', 'hole_3': 'left', 'hole_4': 'right', 'hole_5': 'left', 'hole_6': 'right', 'hole_7': 'top', 'hole_8': 'top'}
stefan_part['part3'] = {'hole_1': 'left', 'hole_2': 'right', 'hole_3': 'left', 'hole_4': 'right', 'hole_5': 'left', 'hole_6': 'right', 'hole_7': 'top', 'hole_8': 'top'}
stefan_part['part4'] = {'hole_1': 'left', 'hole_2': 'right', 'hole_3': 'left', 'hole_4': 'right', 'hole_5': 'left', 'hole_6': 'right', 'hole_7': 'left', 'hole_8': 'right'}
stefan_part['part5'] = {'hole_1': 'top', 'hole_2': 'bottom', 'hole_3': 'top', 'hole_4': 'top', 'hole_5': 'bottom', 'hole_6': 'top', 'hole_7': 'top', 'hole_8': 'top', 'hole_9': 'bottom', 'hole_10': 'top'}
stefan_part['part6'] = {'hole_1': 'bottom', 'hole_2': 'top', 'hole_3': 'bottom', 'hole_4': 'bottom', 'hole_5': 'top', 'hole_6': 'bottom', 'hole_7': 'bottom', 'hole_8': 'bottom', 'hole_9': 'top', 'hole_10': 'bottom'}

# STEFAN INTERMEDIATES(step) HOLE LABELING - faces
stefan_inter = {}
#stefan_inter['step1_a'] = {'part2_1-hole_1': ['left', '101350'], 'part2_1-hole_2': ['right', '101350'], 'part2_1-hole_3': ['left', '104322'], 'part2_1-hole_4': ['right', '104322'], 'part2_1-hole_5': ['left', '101350'], 'part2_1-hole_6': ['right', '101350'], 'C122620_1-hole_1': ['back', '122620'], 'C122620_2-hole_1': ['back', '122620']}
#stefan_inter['step1_b'] = {'part3_1-hole_1': ['left', '101350'], 'part3_1-hole_2': ['right', '101350'], 'part3_1-hole_3': ['left', '104322'], 'part3_1-hole_4': ['right', '104322'], 'part3_1-hole_5': ['left', '101350'], 'part3_1-hole_6': ['right', '101350'], 'C122620_1-hole_1': ['back', '122620'], 'C122620_2-hole_1': ['back', '122620']}
stefan_inter['step1_a'] = {'part2_1-hole_1': 'left', 'part2_1-hole_2': 'right', 'part2_1-hole_3': 'left', 'part2_1-hole_4': 'right', 'part2_1-hole_5': 'left', 'part2_1-hole_6': 'right', 'C122620_1-hole_1': 'back', 'C122620_1-hole_2': 'back'}
stefan_inter['step1_b'] = {'part3_1-hole_1': 'left', 'part3_1-hole_2': 'right', 'part3_1-hole_3': 'left', 'part3_1-hole_4': 'right', 'part3_1-hole_5': 'left', 'part3_1-hole_6': 'right', 'C122620_3-hole_1': 'back', 'C122620_4-hole_1': 'back'}
stefan_inter['step2'] = {'part3_1-hole_4': 'bottom', 'part6_1-hole_1': 'bottom', 'part6_1-hole_2': 'top', 'part6_1-hole_3': 'bottom', 'part6_1-hole_5': 'top', 'part6_1-hole_7': 'bottom', 'part6_1-hole_8': 'bottom', 'part6_1-hole_9': 'top', 'part6_1-hole_10': 'bottom', 'C122620_3-hole_1': 'front', 'C122620_4-hole_1': 'front'}
stefan_inter['step3'] = {'part2_1-hole_2': 'bottom', 'part3_1-hole_4': 'bottom', 'part6_1-hole_2': 'top', 'part6_1-hole_5': 'top', 'part6_1-hole_7': 'bottom', 'part6_1-hole_8': 'bottom', 'part6_1-hole_9': 'top', 'part6_1-hole_10': 'bottom', 'C122620_1-hole_1': 'front', 'C122620_2-hole_1': 'front', 'C122620_3-hole_1': 'front', 'C122620_4-hole_1': 'front'}
stefan_inter['step4'] = {'part2_1-hole_3': 'bottom', 'part3_1-hole_4': 'bottom', 'part4_1-hole_5': 'bottom', 'part6_1-hole_2': 'top', 'part6_1-hole_5': 'top', 'part6_1-hole_9': 'top', 'C122620_1-hole_1': 'front', 'C122620_2-hole_1': 'front', 'C122620_3-hole_1': 'front', 'C122620_4-hole_1': 'front'}
stefan_inter['step5'] = {'part6_1-hole_2': 'top', 'part6_1-hole_5': 'top', 'part6_1-hole_9': 'top', 'C122620_1-hole_1': 'front', 'C122620_2-hole_1': 'front', 'C122620_3-hole_1': 'front', 'C122620_4-hole_1': 'front'}
stefan_inter['step6'] = {'C122620_1-hole_1': 'front', 'C122620_2-hole_1': 'front', 'C122620_3-hole_1': 'front', 'C122620_4-hole_1': 'front'}
stefan_inter['step7'] = {'C122620_1-hole_1': 'front', 'C122620_2-hole_1': 'front', 'C122620_3-hole_1': 'front', 'C122620_4-hole_1': 'front'}
stefan_inter['step8'] = {'C122620_1-hole_1': 'front', 'C122620_2-hole_1': 'front', 'C122620_3-hole_1': 'front', 'C122620_4-hole_1': 'front'}

# pre-defined hole-connectivity 'PART1(related to fasteners)#PART2(may not having fasteners)'
connect = {}
connect['step1_a#step2'] = ['left#part2_1-hole_2#bottom#part6_1-hole_3', 'left#part2_1-hole_6#bottom#part6_1-hole_1']
connect['step1_b#part6'] = ['right#part3_1-hole_1#bottom#hole_6', 'right#part3_1-hole_5#bottom#hole_4']
connect['part4#step3'] = ['right#hole_8#bottom#part6_1-hole_10', 'right#hole_4#bottom#part6_1-hole_8', 'right#hole_2#bottom#part6_1-hole_7']
connect['part5#step4'] = ['bottom#hole_9#bottom#part4_1-hole_5', 'bottom#hole_2#bottom#part2_1-hole_3', 'bottom#hole_5#bottom#part3_1-hole_4']

connector = {}
connector[3.] = '122620'
connector[4.] = '101350'
connector[3.5] = '101350'
connector[2.5] = '104322'
connector[6.3] = '104322'
connector[2.75] = '104322'

############ READ PART JSON FILE
import json
import glob, os
import numpy as np

sort_key= {}
sort_key['top'] = ['x', False, 'y', True]
sort_key['left'] = ['y', True, 'z', True] #'z', False, 'y', True]
sort_key['right'] = ['y', False, 'z', True] #'z', True, 'y', True]
sort_key['bottom'] = ['x', False, 'y', False]
sort_key['front'] = ['x', False, 'z', True]
sort_key['back'] = ['x', True, 'z', True]


def read_hole(json_dir):
    hole_dic = {}
    json_paths = glob.glob(os.path.join(json_dir, '*.json'))
    part_paths = []
    step_paths = []
    connector_paths = []
    for path in json_paths:
        if 'part' in os.path.basename(path):
            part_paths.append(path)
        elif 'step' in os.path.basename(path):
            step_paths.append(path)
        else:
            connector_paths.append(path)

    for path in part_paths:
        # read json file
        part_name = os.path.basename(path).rstrip('.json')
        with open(path, 'r') as f:
            file_data = json.load(f)
        data = file_data["data"]
        if 'hole' not in data.keys():
            hole_dic[part_name] = []
            continue
        if part_name == "part1":
            hole_dic[part_name] = []
            continue
        holes_info = data["hole"]

        part_holes = {}
        for k in holes_info.keys():
            hole_info = holes_info[k]
            name = k
            x,y,z = float(hole_info["CenterX"]), float(hole_info["CenterY"]), float(hole_info["CenterZ"])
            radius = float(hole_info["Radius"])
            con = connector[radius]
            through = hole_info["through"]
            direction = stefan_part[part_name][name]
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
        part_name = os.path.basename(path).rstrip('.json')
        with open(path, 'r') as f:
            file_data = json.load(f)
        data = file_data["data"]
        if "hole" not in data.keys():
            hole_dic[part_name] = []
            continue
        holes_info = data["hole"]
        part_holes = {}
        for k in holes_info.keys():
            hole_info = holes_info[k]
            name = k
            if name in stefan_inter[part_name].keys():
                direction = stefan_inter[part_name.rstrip('.stl')][name]
            else:
#                print(part_name, name, 'not used')
                continue

            x,y,z = float(hole_info["CenterX"]), float(hole_info["CenterY"]), float(hole_info["CenterZ"])
            if 'C122620' in name:
                con = '122925'
                through = False
            else:
                original_name = name.split('-')[0].split('_')[0]
                original_hole_name = 'hole_'+str(int(name.split('-')[1].split('_')[1])) #-1)
                original_hole_info = hole_dic[original_name]
                original_hole_info = [x for x in original_hole_info if x[0] == original_hole_name][0]
                con = original_hole_info[4]
                through = original_hole_info[5]
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

    for k,v in connect.items():
        hole_dic[k] = v

    return hole_dic
