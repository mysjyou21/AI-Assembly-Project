from action import map_action
import numpy as np
import glob
import os
import csv, json
from collections import OrderedDict
import itertools
import shutil

# 은지

def serial_number_to_index(num_dic_filename, OCR_serial_result, OCR_mult_result):
    """ modify OCR_num_result, 192805 to PT0441 """
    # ranch, spanner index
    tool_num = ['100001', '100002', '100006', '100049', '100092', '110470', '111631', '113453', '114463', '128632', '146714', '120202', '108184', '108490']  # CHECK ####

    # read material serial number - part number mapped csv file
    f = open(num_dic_filename, 'r')  # , encoding='utf-8')
    csv_reader = csv.reader(f)
    num_dic = {}
    num_keys = []
    next(csv_reader)
    for line in csv_reader:
        index = line[0]
        for i in range(1, len(line)):
            if line[i] == '':
                break
            num_dic[line[i]] = index
            num_keys.append(line[i])
    f.close()

    # map serial number - part number, assign each part multiplied_number
    OCR_serial_index = []
    OCR_mult_result_mod = []
    # for each cut
    for cut_num in range(len(OCR_serial_result)):
        cut_num_result = OCR_serial_result[cut_num]
        cut_num_index = []
        cut_mult_result = OCR_mult_result[cut_num]
        cut_mult_result_mod = []
        # for each circle
        for circle_num in range(len(cut_num_result)):
            circle_num_result = cut_num_result[circle_num]
            circle_num_index = []
            # assign each circle's mult_result '1' even though it hasn't any number
            if len(cut_mult_result) > 0:
                circle_mult_result = cut_mult_result[circle_num][0]
            else:
                circle_mult_result = '1'
            circle_mult_result_mod = []
            # for each part
            for serial_num in circle_num_result:
                # mult_num
                if circle_num_result.count(serial_num) > 1:  # same serial numbers, several materials
                    serial_mult = str(int(circle_mult_result) * circle_num_result.count(serial_num))
                elif serial_num in tool_num:  # tool always 1 (changing....?)
                    serial_mult = '1'
                else:  # common situation
                    serial_mult = circle_mult_result

                if serial_num in num_dic:
                    serial_idx = num_dic[serial_num]
                else:
                    first_idx = 0
                    max_count = 0
                    for num_key in num_keys:
                        count = 0
                        for i in range(min(len(serial_num), len(num_key))):
                            count += (serial_num[i] == num_key[i])
                        if count > max_count:
                            max_count = count
                            first_idx = num_key
                    serial_idx = num_dic[first_idx]

                circle_num_index.append(serial_idx)
                circle_mult_result_mod.append(serial_mult)
                # circle_num_index = list(set(circle_num_index)) # unordered, removing duplicates
            temp_dic = OrderedDict()
            temp_mult = {}
            for i in range(len(circle_num_index)):
                idx = circle_num_index[i]
                temp_dic[idx] = True
                if idx in temp_mult:
                    temp_mult[idx] = temp_mult[idx] if temp_mult[idx] > circle_mult_result_mod[i] else circle_mult_result_mod[i]
                else:
                    temp_mult[idx] = circle_mult_result_mod[i]

            circle_num_index = list(temp_dic.keys())  # ordered
            cut_num_index.append(circle_num_index)
            cut_mult_result_mod.append(list(temp_mult.values()))
        OCR_serial_index.append(cut_num_index)
        OCR_mult_result_mod.append(cut_mult_result_mod)

    return OCR_serial_index, OCR_mult_result_mod


def write_csv_mission1_1st_year(OCR_serial_index, OCR_mult_result_mod, cut_path, csv_dir):
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    f = open(os.path.join(csv_dir, 'mission1.csv'), 'w', newline='')  # encoding='utf-8', newline='')
    cut_names = sorted(glob.glob(os.path.join(cut_path, '*.png')))
    csv_writer = csv.writer(f)
    csv_writer.writerow(['file_name', 'Label', '#', 'Label', '#', 'Label', '#', 'Label', '#'])
    for cut_num in range(len(OCR_serial_index)):
        cut_name = cut_names[cut_num].split('/')[-1][0:-4]
        cut_row = [cut_name]
        cut_part_lab_ = []
        cut_part_num_ = []
        cut_material = OCR_serial_index[cut_num]
        cut_mult = OCR_mult_result_mod[cut_num]
        for circle_num in range(len(cut_material)):
            circle_material = cut_material[circle_num]
            circle_mult = cut_mult[circle_num]
            for serial_num in range(len(circle_material)):
                serial_material = circle_material[serial_num]
                serial_mult = circle_mult[serial_num]
                cut_part_lab_.append(serial_material)
                cut_part_num_.append(serial_mult)
        cut_part_pair = list(filter(lambda x: x[1] != '0', list(zip(cut_part_lab_, cut_part_num_))))

        cut_part_lab = [x for x, _ in sorted(cut_part_pair)]  # zip(cut_part_lab_, cut_part_num_))]
        cut_part_num = [x for _, x in sorted(cut_part_pair)]  # zip(cut_part_lab_, cut_part_num_))]

        # for dealing with duplicates (across the circles in a cut)
        temp_dic = OrderedDict()
        temp_mult = OrderedDict()
        for i in range(len(cut_part_lab)):
            idx = cut_part_lab[i]
            temp_dic[idx] = True
            if idx in temp_mult:
                temp_mult[idx] += cut_part_num[i]
            else:
                temp_mult[idx] = cut_part_num[i]
        cut_part_lab = list(temp_dic.keys())  # ordered
        cut_part_num = list(temp_mult.values())
        cut_part = list(zip(cut_part_lab, cut_part_num))
        cut_part = list(itertools.chain(*cut_part))

        csv_writer.writerow(cut_row + cut_part)

    f.close()
    print('write mission1 file at ' + os.path.join(csv_dir, 'mission1.csv'))


def write_csv_mission2_1st_year(OCR_serial_index, OCR_mult_result_mod, cut_path, csv_dir):
    """ write csv file for mission2 """
    # output folder
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    # make action dictionary
    f = open(os.path.join('function', 'utilities', 'action_label.csv'), 'r')  # , encoding='utf-8')
    csv_reader = csv.reader(f)
    act_dic = {}   # key: 'PT0001' value: ['0']
    next(csv_reader)  # ignore first line
    for line in csv_reader:
        part_lab = line[0]
        act_dic[part_lab] = line[2:]
    f.close()

    # write csv file
    f = open(os.path.join(csv_dir, 'mission2.csv'), 'w', newline='')  # encoding='utf-8', newline='')
    cut_names = sorted(glob.glob(os.path.join(cut_path, '*.png')))
    csv_writer = csv.writer(f)
    csv_writer.writerow(['File_name', 'Label', '#', 'Label', '#'])
    # for each cut
    for cut_num in range(len(OCR_serial_index)):
        cut_name = cut_names[cut_num].split('/')[-1][0:-4]
        cut_row = [cut_name]
        cut_act_lab = []
        cut_act_num = []
        cut_material = OCR_serial_index[cut_num]
        cut_mult = OCR_mult_result_mod[cut_num]
        for circle_num in range(len(cut_material)):
            circle_material = cut_material[circle_num]
            circle_mult = cut_mult[circle_num]
            circle_action, circle_num = map_action(circle_material, circle_mult, act_dic)
            cut_act_lab += circle_action
            cut_act_num += circle_num

        # process duplicates
        temp_dic = OrderedDict()
        temp_mult = OrderedDict()
        for i in range(len(cut_act_lab)):
            idx = cut_act_lab[i]
            temp_dic[idx] = True
            if idx in temp_mult:
                temp_mult[idx] += cut_act_num[i]
            else:
                temp_mult[idx] = cut_act_num[i]
        cut_act_lab = list(temp_dic.keys())  # ordered
        cut_act_num = list(temp_mult.values())

        cut_act = list(zip(cut_act_lab, cut_act_num))
        cut_act = list(itertools.chain(*cut_act))

        csv_writer.writerow(cut_row + cut_act)
    f.close()

    print('write mission 2 csv file in ' + os.path.join(csv_dir, 'mission2.csv'))


def write_csv_mission(actions, cut_path, step_path, csv_dir):
    """ write contents in actions, in .csv format
        actions: [part1_loc, part1_id, part1_pos, part2_loc, part2_id, part2_pos, connector1_serial_OCR, connector1_mult_OCR, connector2_serial_OCR, connector2_mult_OCR, action_label, is_part1_above_part2(0,1)]"""

    filename = cut_path.split('/')[-2]
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    f = open(os.path.join(csv_dir, 'mission_%s.csv' % step_path), 'w', newline='') #encoding='utf-8'
    cut_names = sorted(glob.glob(os.path.join(cut_path, '*.png')))
    csv_writer = csv.writer(f)
#    csv_writer.writerow(['File_name', 'Label_step_number', 'Label_part_1_1', 'theta', 'phi', 'alpha', '#',
#                         'Label_part_1_2', 'theta', 'phi', 'alpha', '#', 'Label_part_1_3', 'theta', 'phi', 'alpha', '#',
#                         'Label_part_1_4', 'theta', 'phi', 'alpha', '#', 'Label_part_2_1', 'theta', 'phi', 'alpha', '#',
#                         'Label_part_2_2', 'theta', 'phi', 'alpha', '#', 'part_vertical_relation_1', 'part_vertical_relation_2',
#                         '#', 'Label_action_1', '#', 'Label_action_2', '#', 'Label_connector_1', '#', 'Label_connector_2'])
    csv_writer.writerow(['File_name', 'Label_step_number', 'Label_part_1', 'theta', 'phi', 'alpha', 'additional', '#', 'hole',
                         'Label_part_2', 'theta', 'phi', 'alpha', 'additional', '#', 'hole',
                         'Label_part_3', 'theta', 'phi', 'alpha', 'additional', '#', 'hole',
                         'Label_part_4', 'theta', 'phi', 'alpha', 'additional', '#', 'hole',
                         'Label_action', '#', 'Label_connector', '#'])# 1
    for action in actions:
        write_list = [filename, step_path]
        # part
        for i in range(0,4):
            part = action[i]
            if part[0]!='':
                part_loc = part[0] # [x,y,w,h]
                part_id = part[1] # 'string'
                part_pose = part[2] # [theta,phi,alpha]
                part_holes = part[3]
                write_list.append(part_id)
                write_list += part_pose
                write_list.append('1') # part num, default 1
                hole_string = ''
                for part_hole in part_holes:
                    hole_string += part_hole[0]+','
                hole_string = hole_string[0:-1]
                write_list.append(hole_string)
            else:
                write_list += ['', '', '', '', '', '', '']

#        write_list.append('part1up#part2down') # part_vertical_relation
#        try:
        action_lab = action[6]
#        except:
#            action_lab = 'A005'
        write_list.append(action_lab)
        mults = action[5]
        if len(mults)==0:
            mults=['']
        write_list.append(mults[0])
        serials = action[4]
        if len(serials)==0:
            serials=['']
        write_list.append(serials[0])
        write_list.append(mults[0])
#        for i in range(0,2):
#            if i<len(serials):
#                write_list.append(serials[i])
#                write_list.append(mults[i])
#            else:
#                write_list.append('')
#                write_list.append('')
        csv_writer.writerow(write_list)

def write_json_mission(actions, cut_path, step_path, json_dir):
    """ write contents in actions, in json format
        actions: [part1_loc, part1_id, part1_pos, part2_loc, part2_id, part2_pos, connector1_serial_OCR, connector1_mult_OCR, connector2_serial_OCR, connector2_mult_OCR, action_label, is_part1_above_part2(0,1)]"""

    filename = cut_path.split('/')[-2]
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)

    f = open(os.path.join(json_dir, 'mission_%s.json' % step_path), 'w')

    step_dic = OrderedDict()
    step_dic['File_name'] = filename
    step_dic['Label_step_number'] = step_path

    for act_i in range(len(actions)):
        action = actions[act_i]
        write_list = [filename, step_path]
        action_dic = OrderedDict()
        # part
        for i in range(0,4):
            part_dic = OrderedDict()
            part = action[i]
            if part[0]!='':
                part_loc = part[0] # [x,y,w,h]
                part_id = part[1] # 'string'
                part_pose = part[2] # [theta,phi,alpha,additional]
                part_holes = part[3]
                part_dic['label'] = part_id
                part_dic['theta'] = part_pose[0]
                part_dic['phi'] = part_pose[1]
                part_dic['alpha'] = part_pose[2]
                part_dic['additional'] = part_pose[3]
                part_dic['#'] = '1' #default 1
#                hole_string = ''
#                for part_hole in part_holes:
#                    hole_string += part_hole[0]+','
#                hole_string = hole_string[0:-1]
#                write_list.append(hole_string)
#                part_holes = [x[0] for x in part_holes]

                if 'part' in part_id:
                    part_holes = ['%s_1-hole_%s' % (part_id, part_holes[j]) for j in range(len(part_holes))]
                part_holes = sorted(part_holes)
                part_dic['hole'] = part_holes
            action_dic['Part%d' % i] = part_dic

        action_lab = action[6]
        action_lab_dic = OrderedDict()
        action_lab_dic['label'] = action_lab
        mults = action[5]
        if len(mults)==0:
            mults = ['1']
        action_lab_dic['#'] = mults[0]
        action_dic['Action'] = action_lab_dic
        connector_dic = OrderedDict()
        serials = action[4]
        if len(serials)==0:
            serials=['']
        elif len(serials[0])==0:
            mults[0] = '1'
        connector_dic['label'] = 'C'+serials[0] if len(serials[0])>0 else serials[0]
        connector_dic['#'] = mults[0]
        action_dic['Connector'] = connector_dic
        if len(action)==8:
            action_dic['HolePair'] = action[7]
        else:
            action_dic['HolePair'] = []

        step_dic['Action%d' % act_i] = action_dic

    json.dump(step_dic, f, indent=2)

    f.close()
