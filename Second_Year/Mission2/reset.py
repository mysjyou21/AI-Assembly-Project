import os, glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--assembly_name', default='stefan')
parser.add_argument('--input_path', default='./input')
parser.add_argument('--remove_center', type=int, default=1)
opt = parser.parse_args()

input_dir = os.path.join(opt.input_path, opt.assembly_name) #'./input/stefan'
cad_dir = os.path.join(opt.input_path, opt.assembly_name, 'cad') #'./input/stefan/cad'

intermediates = glob.glob(os.path.join(cad_dir, '*.STL'))
for inter in intermediates:
    os.system('mv '+inter+' '+os.path.join(input_dir, os.path.basename(inter)))

print('left ', glob.glob(os.path.join(cad_dir, '*.STL')), glob.glob(os.path.join(cad_dir, '*.obj')))

if opt.remove_center==1:
    center_path = os.path.join(cad_dir, 'center.json')
    if os.path.exists(center_path):
        os.system('rm ' + center_path)
        print('Removed center.json in %s'%(center_path))
