import csv
import numpy as np

with open('./number_labeling.csv') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)
    lab = []
    for line in csv_reader:
        lab.append(line[-8:])
    lab_npy = np.array(lab)
    print(lab_npy.shape)
    np.save('./lab.npy', lab_npy)
