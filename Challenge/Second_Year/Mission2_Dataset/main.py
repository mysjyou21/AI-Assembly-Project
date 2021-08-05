import time
from args import *
from create_dataset import *
from sample_generator import *

def main():
    args = define_args()
    mode = [x for x in args.mode]
    cad_adrs = sorted(glob.glob(args.cad_path + '/*.obj'))
    cad_basenames = [os.path.basename(path) for path in cad_adrs]
    print('\n MODE ' + args.mode + '\n')

    ''' scenes generation '''
    if '1' in mode:
        print('====================================')
        print(' MODE 1 : Scenes Generation')
        print('====================================')
        create_scenes(args)

    ''' Rendering '''
    if '2' in mode:
        print('====================================')
        print(' MODE 2 : Rendering')
        print('====================================')
        create_pointcloud(args, cad_basenames)
        create_rendering(args, cad_basenames)

    ''' Postprocessing'''
    if '3' in mode:
        print('====================================')
        print(' MODE 3 : Postprocessing')
        print('====================================')
        postprocessing(args)

    ''' Create Labels '''
    if '4' in mode:
        print('====================================')
        print(' MODE 4 : Create Labels')
        print('====================================')
        create_labels(args)

    ''' Add non-part components'''
    if '5' in mode:
        print('====================================')
        print(' MODE 5 : Add Non-part Components')
        print('====================================')
        add_non_part_components = Synthesizer(args)
        add_non_part_components.run()


if __name__ == '__main__':
    tic_main = time.time()
    main()
    toc_main = time.time()
    time = toc_main - tic_main
    print('{} hr {} min {} sec'.format(int(time/60/60), int(time/60%60), int(time%60)))


