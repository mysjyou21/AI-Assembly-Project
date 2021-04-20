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


def is_same_original(ref, test, root=False, verbose=False):
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
            try:
                checks.append(is_same_original(ref[k], test[k], False, verbose))
            except:
                checks.append(False)
                print('      no key', k)
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


def is_same_new(ref, test, root=False, is_pose=0, verbose=False, pose_print=False):
    """ ref == test ? """
    if isinstance(ref, dict):
        checks = []
        if is_pose:
            ref_k = sorted(list(ref.keys()).copy())
            test_k = sorted(list(test.keys()).copy())
            matches = []
            for rk in ref_k:
                for tk in test_k:
                    if rk==tk:
                        matches.append([rk,tk])
                        test_k.remove(tk)
                        break
            for rk,tk in matches:
                if test[tk][1] == "1":
                    if pose_print:
                        print("     ", rk, "(pose)")
                    if rk == "part1":
                        if pose_print:
                            print("         **part1 pose is compared with 2 ground truth")
                        checks.append(is_same_new(ref[rk][0].split(",")[0], test[tk][0], 0, verbose, pose_print)\
                        or is_same_new(ref[rk][0].split(",")[1], test[tk][0], 0, verbose, pose_print))
                    else:
                        checks.append(is_same_new(ref[rk][0], test[tk][0], 0, verbose, pose_print))
                else:
                        continue
        else:
            for k in sorted(ref.keys()):
                if root:
                    print("----------file%s----------" % k)
                if k == "sub_part" and len(ref[k].keys())>1:
                    checks.append(is_same_new(ref[k], test[k], False, 1, verbose, pose_print))
                else:
                    if verbose:
                        if "Action" in k and k!="Action":
                            print("  ", k)
                        if "Part" in k:
                            print("    ", k)
                    if k == "part1":
                        checks.append(is_same_new(ref[k][0].split(",")[0], test[k][0], 0, verbose, pose_print)\
                        or is_same_new(ref[k][0].split(",")[1], test[k][0], 0, verbose, pose_print))
                    else:
                        try:
                            checks.append(is_same_new(ref[k], test[k], False, 0, verbose, pose_print))
                        except:
                            checks.append(False)
                            print('      no key', k)
        if False in checks:
            check = False
        else:
            check = True
    elif isinstance(ref, list):
        if sorted(ref)!=sorted(test):
            print('      different list', sorted(ref), sorted(test))
        check = sorted(ref)==sorted(test)
    else:
        if ref!=test:
            print('      different value', ref, test)
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
    parser.add_argument('--pose_print', type=str2bool, default=False)
    args = parser.parse_args()

    # reference(gt) path, test(pred) path
    OUTPUT_PATH = '../../output'
    ref_path = './gt/' + args.assembly_name + '/' + args.compare_type + '_type'
    additional_path = '/pred' if args.compare_type == 'new' else ''
    test_path = OUTPUT_PATH  + '/' + args.assembly_name + additional_path
    

    # load reference
    ref_adrs = sorted(glob.glob(ref_path + '/*.json'))
    if not len(ref_adrs):
        print('Error: no reference files')
    ref_dict = {}
    for ref_adr in ref_adrs:
        with open(ref_adr, 'r') as f:
            step_num = os.path.basename(ref_adr).replace('.json', '').replace('mission_', '')
            ref_dict[step_num] = json.load(f)
            reorder_parts(ref_dict[step_num])

    # load test
    test_adrs = sorted(glob.glob(test_path + '/*.json'))
    if not len(test_adrs):
        print('Error: no test files')
    test_dict = {}
    for test_adr in test_adrs:
        with open(test_adr, 'r') as f:
            step_num = os.path.basename(test_adr).replace('.json', '').replace('mission_', '')
            test_dict[step_num] = json.load(f)
            reorder_parts(test_dict[step_num])

    if args.compare_type == 'original':
        print(is_same_original(ref_dict, test_dict, root=True, verbose=args.verbose))
    else:
        print(is_same_new(ref_dict, test_dict, root=True, verbose=args.verbose, pose_print=args.pose_print))



if __name__ == '__main__':
    main()