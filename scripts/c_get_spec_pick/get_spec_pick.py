import numpy as np
import os


def process_lab_txt(path_txt):
    with open(path_txt, 'r') as file:
        curve_data = eval(file.read())  
    for key, value in curve_data.items():
        curve_data[key] = np.array([list(map(int, point)) for point in value])
    return curve_data

for field in ['bp', 'yz', 'sh']:
    print('# ------ %s ------' % field)
    root_path = '/data/wht/seismic/RCP/%s/pick_txt' % field
    files = sorted(os.listdir(root_path))
    print('There are %d samples' % len(files))
    curve_list = []
    for file_n in files:
        curve_spec = process_lab_txt(os.path.join(root_path, file_n))
        
        if curve_spec:
            curve_list.append(file_n)
    print('There are %d samples with spec pick' % len(curve_list))
    print('Head 20 samples', curve_list[:20])

