import json
import numpy as np

def mid_loader(step_name, json_path, scale=1):
    """ input: step_name = 'step3'
        output: hole_XYZ(np array),
                hole_dic={hole_name: [CenterX, CenterY, CenterZ]}
                hole_name: 'part2_1-hole_3' """

    part_file = '%s/%s.json'%(json_path, step_name)
    part_data = []
    with open(part_file, 'r') as f:
        part_data = json.load(f)

    part_dic = part_data["data"]
    holes = part_dic["hole"]
    holename = [x for x in sorted(holes.keys())]
    id_list = list(set([hole.split('-')[0] for hole in holename]))
    id_list = [x.replace('_1','') if 'C122620' not in x else 'part8' if (x=='C122620_1' or x=='C122620_2') else 'part7' for x in id_list]
    id_list = list(set(id_list))

    hole_XYZ = [[holes[k]["CenterX"], holes[k]["CenterY"], holes[k]["CenterZ"]] for k in holename]
    if "CenterPointX" in part_dic.keys():
        center_XYZ = [part_dic["CenterPointX"], part_dic["CenterPointY"], part_dic["CenterPointZ"]]
    else:
        center_XYZ = [0,0,0]

    center_XYZ = np.array(list(map(float, center_XYZ)))/scale
    hole_XYZ = [np.array(list(map(float, temp)))/scale for temp in hole_XYZ]

    hole_XYZ = [hole_XYZ[i] for i in range(len(hole_XYZ))]
    hole_XYZ = np.stack(hole_XYZ)

    part_hole_idx = np.array([hole.split('-')[0] for hole in holename])
    part_hole_idx = np.array([x.replace('_1','') if 'C122620' not in x else 'part8' if (x=='C122620_1' or x=='C122620_2') else 'part7' for x in part_hole_idx])

    part_hole_dic = {}
    for id in id_list:
        idx = np.where(part_hole_idx == id)[0]
        part_hole_XYZ = hole_XYZ[idx]
        part_hole_dic[id] = part_hole_XYZ

    if 'part7' in id_list:
        assert 'part2' in id_list
        part_hole_dic['part7'] = np.concatenate([part_hole_dic['part7'], part_hole_dic['part2']])
        del part_hole_dic['part2']
    if 'part8' in id_list:
        assert 'part3' in id_list
        part_hole_dic['part8'] = np.concatenate([part_hole_dic['part8'], part_hole_dic['part3']])
        del part_hole_dic['part3']

    return part_hole_dic


def base_loader(part_name, json_path, scale=1):
    """ input: part_name = 'part3'
        output: hole_XYZ(np array), #norm_XYZ(np array),
                hole_dic={hole_name: [CenterX, CenterY, CenterZ]} #, NormalX, NormalY, NormalZ]}
                hole_name: 'part2_1-hole_3' """
    if part_name == "part7":
        part_name = "step1_b"
    elif part_name == "part8":
        part_name = "step1_a"

    part_file = '%s/%s.json'%(json_path, part_name)
    part_data = []
    with open(part_file, 'r') as f:
        part_data = json.load(f)

    part_dic = part_data["data"]
    holes = part_dic["hole"]
    holename = [x for x in sorted(holes.keys())]
    if part_name == 'step1_a' or part_name == 'step1_b':
        holename1 = sorted([x for x in holename if 'C122620' in x])
        holename2 = sorted([x for x in holename if 'C122620' not in x], key=lambda x: int(x.split('-')[1].split('_')[1]))
        holename = holename2 + holename1
    else:
        holename = sorted(holename, key=lambda x:int(x.split('_')[1]))
    hole_XYZ = [[holes[k]["CenterX"], holes[k]["CenterY"], holes[k]["CenterZ"]] for k in holename]
    if "CenterPointX" in part_dic.keys():
        center_XYZ = [part_dic["CenterPointX"], part_dic["CenterPointY"], part_dic["CenterPointZ"]]
    else:
        center_XYZ = [0,0,0]

    center_XYZ = np.array(list(map(float, center_XYZ)))/scale
    hole_XYZ = [np.array(list(map(float, temp)))/scale for temp in hole_XYZ]

    hole_XYZ = [hole_XYZ[i] for i in range(len(hole_XYZ))]
    hole_XYZ = np.stack(hole_XYZ)

    return hole_XYZ


