import json
import numpy as np
import os, glob

# part7 = step1_b = part2 + C122620_3, C122620_4
# part8 = step1_a = part3 + C122620_1, C122620_2

def mid_loader(step_name, json_path, center_dir, scale=100, used_parts=[]):
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

    if len([x for x in id_list if 'C122620' in x])>0: # part7 or part8 exist
        ### to check! assume that part7,8 = step1*
        if 'cad_info2' in json_path: #TO CHECK FALSE PART7 or 8, in cad_info2
            json_path_ref = json_path.replace('cad_info2', 'cad_info')
            with open('%s/%s.json'%(json_path_ref, step_name), 'r') as f:
                part_data_ref = json.load(f)
            id_list_ref = list(set([x.split('-')[0] for x in part_data_ref["data"]["hole"].keys()]))
        else: # in cad_info
            json_path_ref = json_path

        json_paths_ref_step1 = glob.glob('%s/step1*.json'%(json_path_ref))
        json_paths_ref_step1 = sorted(json_paths_ref_step1)

        ids_step1 = []
        C122620_lists = []
        for step1_path in json_paths_ref_step1:
            with open(step1_path, 'r') as f:
                part_data_step1 = json.load(f)
            id_list_step1 = list(set([x.split('-')[0] for x in part_data_step1["data"]["hole"].keys()]))
            C122620_list = [x for x in id_list_step1 if 'C122620' == x.split('_')[0]]
            ids_step1.append(id_list_step1)
            C122620_lists.append(C122620_list)
        ###

        if 'part2_1' in id_list or 'part2_2' in id_list:
            if len(ids_step1) > 1: # 'step1_a.json', 'step1_b.json' --> part7_1, part8_1 OR part7_1, part7_2
                for id_list_step1 in ids_step1:
                    if 'part2_1' in id_list_step1 or 'part2_2' in id_list_step1:
                        part7_exist = 1
            else: # only 'step1.json' --> part7_1 OR part8_1
                if 'part2_1' in ids_step1[0]:
                    part7_exist = 1
        if 'part3_1' in id_list or 'part3_2' in id_list:
            if len(ids_step1) > 1: # 'step1_a.json', 'step1_b.json' --> part7_1, part8_1 OR part8_1, part8_2
                for id_list_step1 in ids_step1:
                    if 'part3_1' in id_list_step1 or 'part3_2' in id_list_step1:
                        part8_exist = 1; break
            else: # only 'step1.json' --> part7_1 OR part8_1
                if 'part3_1' in ids_step1[0]:
                    part8_exist = 1
        if not part7_exist and not part8_exist:
            part7_exist = 1 if 7 in used_parts else 0
            part8_exist = 1 if 8 in used_parts else 0
        if (part7_exist and part8_exist) and (not os.path.exists('%s/%s.json'%(json_path, 'step1_a'))): # FALSE PART7, PART8 ## Assume step1.json means one part, step1_a.json and step1_b.json mean two part(part7, part8), so need to check the situation where step1_a.json and step1_b.json mean TWO part7 or TWO part8 // We may not need this part..
            print('False-part7,8 in mid-hole-loader')

    if part7_exist and part8_exist: # ONLY TRUE PART7,8
        with open('%s/%s.json'%(json_path, 'step1_b'), 'r') as f:
            part7_data = json.load(f)['data']
        with open('%s/%s.json'%(json_path, 'step1_a'), 'r') as f:
            part8_data = json.load(f)['data']

        part7_1_Cind = [x.split('-')[0] for x in part7_data['hole'].keys() if 'C122620' in x]
        part7_1_Cind = list(set(part7_1_Cind))
        part8_1_Cind = [x.split('-')[0] for x in part8_data['hole'].keys() if 'C122620' in x]
        part8_1_Cind = list(set(part8_1_Cind))

#        id_list = [x.replace('_1','') if 'C122620' not in x else 'part7' if x in part7_1_Cind else 'part8' for x in id_list] ## only ONE set
        id_list = [x if 'C122620' not in x else 'part7_1' if x in part7_1_Cind else 'part8_1' for x in id_list]
    elif part7_exist or part8_exist: # PART7 or PART8 or FP PART7/8
        amb_id = 'part7_1' if part7_exist else 'part8_1'
#        id_list = [x.replace('_1','') if 'C122620' not in x else amb_id for x in id_list] ## only ONE set

        if len(ids_step1) == 1:
            id_list = [x if 'C122620' not in x else amb_id for x in id_list]
        else: # len = 2
            if ('part3_1' in ids_step1[0] and 'part2_1' in ids_step1[1]) or ('part3_1' in ids_step1[1] and 'part2_1' in ids_step1[0]): # PART7 or PART8
                id_list = [x if 'C122620' not in x else amb_id for x in id_list]
            else: # part7_1, part7_2 or part8_1, part8_2
                if 'part7' in amb_id:
                    part7_1_Cind = [x.split('-')[0] for x in ids_step1[0] if 'C122620' in x]
                    part7_2_Cind = [x.split('-')[0] for x in ids_step1[1] if 'C122620' in x]
                    id_list = [x if 'C122620' not in x else amb_id if x in part7_1_Cind else amb_id.replace('_1','_2') for x in id_list]
                else:
                    part8_1_Cind = [x.split('-')[0] for x in ids_step1[0] if 'C122620' in x]
                    part8_2_Cind = [x.split('-')[0] for x in ids_step1[1] if 'C122620' in x]
                    id_list = [x if 'C122620' not in x else amb_id if x in part8_1_Cind else amb_id.replace('_1','_2') for x in id_list]
#    else: ## only ONE set
#        id_list = [x.replace('_1','') for x in id_list] ## only ONE set
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
#        part_hole_idx = np.array([x.replace('_1','') if 'C122620' not in x else 'part7' if x in part7_1_Cind else 'part8' for x in part_hole_idx]) ## only ONE set
        part_hole_idx = np.array([x if 'C122620' not in x else 'part7_1' if x in part7_1_Cind else 'part8_1' for x in part_hole_idx])
    elif part7_exist or part8_exist:
#        part_hole_idx = np.array([x.replace('_1','') if 'C122620' not in x else amb_id for x in part_hole_idx]) ## only ONE set
        if len(ids_step1) == 1:
            part_hole_idx = np.array([x if 'C122620' not in x else amb_id for x in part_hole_idx])
        else:
            if ('part3_1' in ids_step1[0] and 'part2_1' in ids_step1[1]) or ('part3_1' in ids_step1[1] and 'part2_1' in ids_step1[0]): # PART7 or PART8
                part_hole_idx = np.array([x if 'C122620' not in x else amb_id for x in part_hole_idx])
            else:
                if 'part7' in amb_id:
                    part_hole_idx = np.array([x if 'C122620' not in x else amb_id if x in part7_1_Cind else amb_id.replace('_1', '_2') for x in part_hole_idx])
                else:
                    part_hole_idx = np.array([x if 'C122620' not in x else amb_id if x in part8_1_Cind else amb_id.replace('_1', '_2') for x in part_hole_idx])
#    else: ## only ONE set
#        part_hole_idx = np.array([x.replace('_1','') for x in part_hole_idx]) ## only ONE set
    ###########################################

    part_hole_dic = {}
    part_holename_dic = {}
    for id in id_list:
        idx = np.where(part_hole_idx == id)[0]
#        print(id, ':', ','.join(list(map(holename.__getitem__, idx)))) # debug
        part_hole_XYZ = hole_XYZ[idx]
        part_hole_dic[id] = part_hole_XYZ
        part_holename_dic[id] = list(map(holename.__getitem__, idx))

    if 'part7_1' in id_list:
#        assert 'part2' in id_list, "part7 has to contain holes with name part2_1-hole_#" - for cad_info, not for cad_info2
        if 'part2_1' in part_hole_dic.keys():
            part_hole_dic['part7_1'] = np.concatenate([part_hole_dic['part2_1'], part_hole_dic['part7_1']])
            part_holename_dic['part7_1'] = part_holename_dic['part2_1'] + part_holename_dic['part7_1']
            del part_hole_dic['part2_1']
            del part_holename_dic['part2_1']
    if 'part7_2' in id_list:
#        assert 'part2' in id_list, "part7 has to contain holes with name part2_2-hole_#" - for cad_info, not for cad_info2
        if 'part2_2' in part_hole_dic.keys():
            part_hole_dic['part7_2'] = np.concatenate([part_hole_dic['part2_2'], part_hole_dic['part7_2']])
            part_holename_dic['part7_2'] = part_holename_dic['part2_2'] + part_holename_dic['part7_2']
            del part_hole_dic['part2_2']
            del part_holename_dic['part2_2']
    if 'part8_1' in id_list:
#        assert 'part3' in id_list, "part8 has to contain holes with name part3_1-hole_#" - for cad_info, not for cad_info2
        if 'part3_1' in part_hole_dic.keys():
            part_hole_dic['part8_1'] = np.concatenate([part_hole_dic['part3_1'], part_hole_dic['part8_1']])
            part_holename_dic['part8_1'] = part_holename_dic['part3_1'] + part_holename_dic['part8_1']
            del part_hole_dic['part3_1']
            del part_holename_dic['part3_1']
    if 'part8_2' in id_list:
#        assert 'part3' in id_list, "part8 has to contain holes with name part3_2-hole_#" - for cad_info, not for cad_info2
        if 'part3_2' in part_hole_dic.keys():
            part_hole_dic['part8_2'] = np.concatenate([part_hole_dic['part3_2'], part_hole_dic['part8_2']])
            part_holename_dic['part8_2'] = part_holename_dic['part3_2'] + part_holename_dic['part8_2']
            del part_hole_dic['part3_2']
            del part_holename_dic['part3_2']
    return part_hole_dic, part_holename_dic


def base_loader(part_name, json_path, center_dir, scale=100):
    """ input: part_name = 'part3_1'
        output: hole_XYZ(np array), #norm_XYZ(np array),
                hole_dic={hole_name: [CenterX, CenterY, CenterZ]} #, NormalX, NormalY, NormalZ]}
                hole_name: 'part2_1-hole_3' """
    ############# TO DEAL WITH PART7, PART8###
    if "part7" in part_name or "part8" in part_name:
        check_files = [os.path.basename(x).replace('.json','') for x in glob.glob('%s/step1*.json'%(json_path))]
        if len(check_files) == 1 and check_files[0] == "step1": # only part7_1 or part8_1
            part_name = "step1"
        elif "step1_a" in check_files and "step1_b" in check_files: # part7_1, part8_1 OR part7_1, part7_2 OR part8_1, part8_2
#            part_name = "step1_b" if "part7" in part_name else "step1_a"
            with open('%s/%s.json'%(json_path, 'step1_a'), 'r') as f:
                check_step1_a = json.load(f)
                check_step1_a = check_step1_a['data']['hole'].keys()
                check_step1_a = [x.split('-')[0] for x in check_step1_a]
            with open('%s/%s.json'%(json_path, 'step1_b'), 'r') as f:
                check_step1_b = json.load(f)
                check_step1_b = check_step1_b['data']['hole'].keys()
                check_step1_b = [x.split('-')[0] for x in check_step1_b]
            if 'part2_1' in check_step1_b and 'part3_1' in check_step1_a:
                part_name = "step1_b" if "part7" in part_name else "step1_a"
            else:
                part_name = "step1_a" if "_1" in part_name else "step1_b" ## assume 'part2_1' -> 'step1_a', 'part2_2' -> 'step1_b'
        else:
            print("step 1 error......, no step1*.json files in %s"%(json_path))
    ###########################################

    if "step" in part_name:
        part_file = '%s/%s.json'%(json_path, part_name)
    else:
        part_file = '%s/%s.json'%(json_path, part_name.split('_')[0])
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
    if 'step' in part_name:
        if part_name in center_data.keys():
            center_XYZ = center_data[part_name]
        else:
            print('No center data, ', part_name)
            center_XYZ = [0,0,0]

    else:
        if part_name.split('_')[0] in center_data.keys():
            center_XYZ = center_data[part_name.split('_')[0]]
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
        min_blender_XYZ = np.array(min_data[part_name.split('_')[0]])/scale
        min_XYZ = min_XYZ - min_blender_XYZ

    hole_XYZ = [np.array(list(map(float, temp)))/scale - center_XYZ - min_XYZ for temp in hole_XYZ]

    hole_XYZ = [hole_XYZ[i] for i in range(len(hole_XYZ))]
    hole_XYZ = np.stack(hole_XYZ)

    return hole_XYZ, holename


