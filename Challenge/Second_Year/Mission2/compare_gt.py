import json
import glob, os
import argparse

def str2bool(string):
    if type(string) == bool:
        bool_type = string
    else:
        assert string == 'True' or string == 'False'
        bool_type = True if string == 'True' else False
    return bool_type


def is_same(ref, test, root=False, verbose=False):
    """ Compare elements recursively. print if elements are different. """
    if isinstance(ref, dict):
        checks = []
        for k in sorted(ref.keys()):
            if root:
                print("----------file%s----------"%k)
            if verbose:
                if "Action" in k and k!="Action":
                    print("  ", k)
                if "Part" in k:
                    print("    ", k)
            checks.append(is_same(ref[k], test[k], False, verbose))
        if False in checks:
            check = False
        else:
            check = True
    elif isinstance(ref, list):
        if sorted(ref)!=sorted(test):
            print('        different', sorted(ref), sorted(test))
        check = sorted(ref)==sorted(test)
    else:
        if ref!=test:
            print('        different', ref, test)
        check = ref==test
    return check


def reorder_parts(dictionary):
    """ sort mission_# dict Part# by part names(labels) """
    for k0, v0 in sorted(dictionary.items()):
        if 'Action' in k0:
            # is Action dict
            unsorted_parts = []
            for k1, v1 in sorted(v0.items()):
                if 'Part' in k1:
                    try:
                        part_name = v1['label']
                        unsorted_parts.append([part_name, v1])
                    except:
                        pass
            sorted_parts = sorted(unsorted_parts)
            if len(sorted_parts):
                for i in range(len(sorted_parts)):
                    _, part_dict = sorted_parts[i]
                    dictionary[k0]["Part%s"%i] = part_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--assembly_name', default='stefan')
    parser.add_argument('--compare_type', default='original', help='original : original, new : parts pose -> intermediate pose')
    parser.add_argument('--verbose', type=str2bool, default=False)
    args = parser.parse_args()

    OUTPUT_PATH = '../../output'
    ref_path = './' + args.assembly_name + '/' + args.compare_type + '_type'
    test_path = OUTPUT_PATH  + '/' + args.assembly_name
    

    # load reference
    ref_adrs = sorted(glob.glob(ref_path + '/*.json'))
    ref_dict = {}
    for ref_adr in ref_adrs:
        with open(ref_adr, 'r') as f:
            step_num = os.path.basename(ref_adr).replace('.json', '').replace('mission_', '')
            ref_dict[step_num] = json.load(f)
            reorder_parts(ref_dict[step_num])

    # load test
    test_adrs = sorted(glob.glob(test_path + '/*.json'))
    test_dict = {}
    for test_adr in test_adrs:
        with open(test_adr, 'r') as f:
            step_num = os.path.basename(test_adr).replace('.json', '').replace('mission_', '')
            test_dict[step_num] = json.load(f)
            reorder_parts(test_dict[step_num])
            
    print(is_same(ref_dict, test_dict, True, args.verbose))

if __name__ == '__main__':
    main()