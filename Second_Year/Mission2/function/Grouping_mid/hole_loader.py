import json
import numpy as np
import os, glob

# part7 = step1_b = part2 + C122620_3, C122620_4
# part8 = step1_a = part3 + C122620_1, C122620_2

def mid_loader(step_name, json_path, center_dir, scale=100):
    """ input: step_name = 'step3'
        output: hole_XYZ(np array),
                hole_dic={hole_name: [CenterX, CenterY, CenterZ]}
                hole_name: 'part2_1-hole_3' """

    part_file = '%s/%s.json'%(json_path, step_name)
    part_data = []
    with open(part_file, 'r') as f:
        part_data = json.load(f)
    with open(os.path.join(center_dir, 'center.json'), 'r') as f:
        center_data = json.load(f)

    part_dic = part_data["data"]
    holes = part_dic["hole"]
    holename = [x for x in sorted(holes.keys())]
    holename1 = sorted(sorted([x for x in holename if 'C122620' not in x], key=lambda x:int(x.split('-')[1].split('_')[1])), key=lambda x: x.split('-')[0]) #key=lambda x:int(x.split('-')[0].split('_')[1]))
    holename2 = sorted(sorted([x for x in holename if 'C122620' in x], key=lambda x:int(x.split('-')[1].split('_')[1])), key=lambda x:int(x.split('-')[0].split('_')[1]))
    holename = holename1 + holename2
#    print(step_name, '::', ', '.join(holename)) #debug
    id_list = list(set([hole.split('-')[0] for hole in holename]))
    ############# TO DEAL WITH PART7, PART8###
    # determine 'part7'(part7_exist=1) in id_list, 'part8'(part8_exist=1) in id_list
    part7_exist = 0
    part8_exist = 0
    if 'part2_1' in id_list and len([x for x in id_list if 'C122620' in x])>0:
        part7_exist = 1
    if 'part3_1' in id_list and len([x for x in id_list if 'C122620' in x])>0:
        part8_exist = 1
    if part7_exist and part8_exist:
        with open('%s/%s.json'%(json_path, 'step1_b'), 'r') as f:
            part7_data = json.load(f)['data']
        with open('%s/%s.json'%(json_path, 'step1_a'), 'r') as f:
            part8_data = json.load(f)['data']

        part7_Cind = [x.split('-')[0] for x in part7_data['hole'].keys() if 'C122620' in x]
        part7_Cind = list(set(part7_Cind))
        part8_Cind = [x.split('-')[0] for x in part8_data['hole'].keys() if 'C122620' in x]
        part8_Cind = list(set(part7_Cind))

        id_list = [x.replace('_1','') if 'C122620' not in x else 'part7' if x in part7_Cind else 'part8' for x in id_list]
    elif part7_exist or part8_exist:
        amb_id = 'part7' if part7_exist else 'part8'
        id_list = [x.replace('_1','') if 'C122620' not in x else amb_id for x in id_list]
    else:
        id_list = [x.replace('_1','') for x in id_list]
    ###########################################
    id_list = list(set(id_list))

    hole_XYZ = [[holes[k]["CenterX"], holes[k]["CenterY"], holes[k]["CenterZ"]] for k in holename]
    if step_name in center_data.keys():
        center_XYZ = center_data[step_name]
    else:
        print('No center data, ', step_name)
        center_XYZ = [0,0,0]
#        center_XYZ = [part_dic["CenterPointX"], part_dic["CenterPointY"], part_dic["CenterPointZ"]]
#        min_XYZ = [part_dic["MinPointX"], part_dic["MinPointY"], part_dic["MinPointZ"]]
    min_XYZ = [0,0,0]

    center_XYZ = np.array(list(map(float, center_XYZ)))/scale
    min_XYZ = np.array(list(map(float, min_XYZ)))/scale
    hole_XYZ = [np.array(list(map(float, temp)))/scale - center_XYZ - min_XYZ for temp in hole_XYZ]

    hole_XYZ = [hole_XYZ[i] for i in range(len(hole_XYZ))]
    hole_XYZ = np.stack(hole_XYZ)

    part_hole_idx = np.array([hole.split('-')[0] for hole in holename])
    ############# TO DEAL WITH PART7, PART8###
    if part7_exist and part8_exist:
        part_hole_idx = np.array([x.replace('_1','') if 'C122620' not in x else 'part7' if x in part7_Cind else 'part8' for x in part_hole_idx])
    elif part7_exist or part8_exist:
        part_hole_idx = np.array([x.replace('_1','') if 'C122620' not in x else amb_id for x in part_hole_idx])
    else:
        part_hole_idx = np.array([x.replace('_1','') for x in part_hole_idx])
    ###########################################

    part_hole_dic = {}
    for id in id_list:
        idx = np.where(part_hole_idx == id)[0]
#        print(id, ':', ','.join(list(map(holename.__getitem__, idx)))) # debug
        part_hole_XYZ = hole_XYZ[idx]
        part_hole_dic[id] = part_hole_XYZ

    if 'part7' in id_list:
        assert 'part2' in id_list, "part7 has to contain holes with name part2_1-hole_#"
        part_hole_dic['part7'] = np.concatenate([part_hole_dic['part2'], part_hole_dic['part7']])
        del part_hole_dic['part2']
    if 'part8' in id_list:
        assert 'part3' in id_list, "part8 has to contain holes with name part3_1-hole_#"
        part_hole_dic['part8'] = np.concatenate([part_hole_dic['part3'], part_hole_dic['part8']])
        del part_hole_dic['part3']

    return part_hole_dic


def base_loader(part_name, json_path, center_dir, scale=100):
    """ input: part_name = 'part3'
        output: hole_XYZ(np array), #norm_XYZ(np array),
                hole_dic={hole_name: [CenterX, CenterY, CenterZ]} #, NormalX, NormalY, NormalZ]}
                hole_name: 'part2_1-hole_3' """
    ############# TO DEAL WITH PART7, PART8###
    if part_name == "part7" or part_name == "part8":
        check_files = [os.path.basename(x).replace('.json','') for x in glob.glob('%s/step1*.json'%(json_path))]
        if len(check_files) == 1 and check_files[0] == "step1":
            part_name = "step1"
        elif "step1_a" in check_files and "step1_b" in check_files:
            part_name = "step1_b" if part_name=="part7" else "step1_a"
        else:
            print("step 1 error......, no step1*.json files in %s"%(json_path))
    ###########################################

    part_file = '%s/%s.json'%(json_path, part_name)
    part_data = []
    with open(part_file, 'r') as f:
        part_data = json.load(f)
    with open(os.path.join(center_dir, 'center.json'), 'r') as f:
        center_data = json.load(f)
    with open(os.path.join(center_dir, 'obj_min.json'), 'r') as f: # to correct global coordinates of our obj files
        min_data = json.load(f)

    part_dic = part_data["data"]
    holes = part_dic["hole"]
    holename = [x for x in sorted(holes.keys())]
    if 'step1' in part_name:
        holename1 = sorted([x for x in holename if 'C122620' not in x], key=lambda x:int(x.split('-')[1].split('_')[1]))
        holename2 = sorted(sorted([x for x in holename if 'C122620' in x], key=lambda x:int(x.split('-')[1].split('_')[1])), key=lambda x:int(x.split('-')[0].split('_')[1]))
        holename = holename1 + holename2
    else:
        holename = sorted(holename, key=lambda x:int(x.split('_')[1]))
#    print(part_name, '::', ', '.join(holename)) # debug
    hole_XYZ = [[holes[k]["CenterX"], holes[k]["CenterY"], holes[k]["CenterZ"]] for k in holename]
    if part_name in center_data.keys():
        center_XYZ = center_data[part_name]
    else:
        print('No center data, ', part_name)
        center_XYZ = [0,0,0]
#        center_XYZ = [part_dic["CenterPointX"], part_dic["CenterPointY"], part_dic["CenterPointZ"]]
    if "MinPointX" in part_dic.keys() and 'step' not in part_name:
        min_XYZ = [part_dic["MinPointX"], part_dic["MinPointY"], part_dic["MinPointZ"]]
    else:
        min_XYZ = [0,0,0]

    center_XYZ = np.array(list(map(float, center_XYZ)))/scale
    min_XYZ = np.array(list(map(float, min_XYZ)))/scale
    if 'step' not in part_name:
        min_blender_XYZ = np.array(min_data[part_name])/scale
        min_XYZ = min_XYZ - min_blender_XYZ

    hole_XYZ = [np.array(list(map(float, temp)))/scale - center_XYZ - min_XYZ for temp in hole_XYZ]

    hole_XYZ = [hole_XYZ[i] for i in range(len(hole_XYZ))]
    hole_XYZ = np.stack(hole_XYZ)

    return hole_XYZ


