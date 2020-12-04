import os, glob

input_dir = './input/stefan'
cad_dir = './input/stefan/cad'

intermediates = glob.glob(os.path.join(cad_dir, '*.STL'))
for inter in intermediates:
    os.system('mv '+inter+' '+os.path.join(input_dir, os.path.basename(inter)))

print('left ', glob.glob(os.path.join(cad_dir, '*.STL')), glob.glob(os.path.join(cad_dir, '*.obj')))

