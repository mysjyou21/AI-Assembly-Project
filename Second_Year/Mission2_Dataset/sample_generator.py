import numpy as np
import matplotlib.pyplot as plt
import glob
import json
import os
import random
from skimage.transform import rescale

class Synthesizer():
    def __init__(self, args):
        self.H = 2480
        self.W = 1748
        self.max_comp_per_img = 4
        self.N_data = 1000
        self.count = 0
        self.augment_per_image = 1
        self.scale = [0.1, 0.95]
        self.cut_num_loc = [220, 191]
        self.bottom_num_loc = [[1575, 2336, 20, 32],[158, 2331, 22, 35]]
        self.location_range={
                'TL' : [300,  0,    600,   100],
                'TR' : [1000, 0,    1100,  100],
                'ML' : [0,  500,  100,  800],
                'MM' : [400,  500,  800,  800],
                'MR' : [1100,  500,  1200, 800],
                'BL' : [100,  1600, 200,  1800],
                #'BM' : [400, 1600, 800,  1800],
                'BR' : [1000, 1600, 1100, 1800]
                }
        self.json = self.load_json('./' + args.label_json + '.json')
        self.comps = self.load_components(args.masked_components_path)
        self.cut_numbers = self.load_numbers(args.cut_num_path)
        self.bottom_numbers = self.load_numbers(args.bottom_num_path)
        self.output_path = args.output_path
    
    def run(self):
        self.main()


    def load_json(self, json_file):
        with open(json_file) as file:
            json_data = json.load(file)
        return json_data


    def load_components(self, path):
        comp_list = glob.glob('%s/*/*.png'%path)
        comps = {}
        for location in self.location_range.keys():
            comp_list = glob.glob('%s/%s/*.png'%(path, location))
            comps[location] = [plt.imread(a) for a in comp_list]
        return comps

    def load_numbers(self, path):
        num_list = glob.glob('%s/*.png'%path)
        numbers = [plt.imread(a) for a in num_list]
        return numbers


    def synthesize_cut(self, assembly_image, N_comp):
        #wood_mask = np.zeros([self.H, self.W])
        #for wood_block in wood_block_info:
        #    [mask_x, mask_y, mask_w, mask_h] = wood_block['bbox_obj']
        #    wood_mask[mask_y : mask_y+mask_h, mask_x : mask_x+mask_w] = 1

        loc_idx = random.sample(list(self.comps.keys()), N_comp)
        for loc in loc_idx:
            redraw = True
            while redraw:
                loc_range = self.location_range[loc]
                x = np.random.randint(loc_range[0], loc_range[2])
                y = np.random.randint(loc_range[1], loc_range[3])
                s = np.random.rand(1) * self.scale[0] + self.scale[1]
                comp = random.choice(self.comps[loc])
                comp = rescale(comp, s, anti_aliasing=True)
                if x+comp.shape[1] < self.W and y+comp.shape[0] < self.H:
                    redraw=False
            assembly_image[y:y+comp.shape[0], x:x+comp.shape[1]] *=\
                    1-comp[...,3,np.newaxis]
            assembly_image[y:y+comp.shape[0], x:x+comp.shape[1]] +=\
                    comp[...,3,np.newaxis]*comp[...,:3]
            assembly_image = np.clip(assembly_image, 0, 1)
        return assembly_image

    def draw_number(self, assembly_image, numbers, loc):
        x = loc[0]
        y = loc[1]
        number = random.choice(numbers)
        assembly_image[y:y+number.shape[0], x:x+number.shape[1]] = number[..., :3]
        return assembly_image


    def main(self):
        for k in range(self.augment_per_image):
            for cut in sorted(self.json.keys()):
                for view in sorted(self.json[cut].keys()):
                    os.makedirs(self.output_path + '/%s/synth_result/%s'%(cut, view), exist_ok=True)
                    assembly_image = plt.imread(self.output_path + '/%s/rgb/%s.png'%(cut, view))
                    
                    iter = np.random.randint(2, self.max_comp_per_img)
                    assembly_image = self.synthesize_cut(assembly_image, iter)

                    assembly_image = self.draw_number(
                            assembly_image, self.cut_numbers, self.cut_num_loc)

                    bottom_num_loc = random.choice(self.bottom_num_loc)
                    assembly_image = self.draw_number(
                            assembly_image, self.bottom_numbers, bottom_num_loc)

                    plt.imsave(self.output_path + '/%s/synth_result/%s/%i.png'%(cut, view, self.count),
                            assembly_image)
                    print('cut:%s view:%s'%(cut, view))
                    self.count+=1

        print('done')

        
if __name__=='__main__':
    add_non_part_components = Synthesizer()
    add_non_part_components.run()
