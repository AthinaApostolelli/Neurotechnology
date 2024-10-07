import os
import numpy as np
from getDeadChannels import find_highImp_channels

animals = ['rEO_06']
basepath = 'D:\Rat_Recording'
sessname = '9_240229_163045'
threshold = [100000, 2000000]

for animal in animals:   
    # Get dead channels
    if animal == 'rEO_06':
        impedance_files = [os.path.join(basepath, animal,'9_impedance.csv')] 

    for session in impedance_files:
        dead_left, dead_right = find_highImp_channels(session, threshold)

    # Get site map
    with open(os.path.join(basepath, animal, sessname, 'amplifier.prm'), 'r') as file:
        prm_content = file.readlines()

    for line in prm_content:
        if 'siteMap' in line:
            # Extract siteMap information
            siteMap_str = line.split('=')[1].split(';')[0].strip()
            siteMap = [int(val) for val in siteMap_str[1:-1].split(',')]

            dead_idx = []
            dead_idx.extend([idx + 64 for idx in dead_left.index.tolist()])
            dead_idx.extend(dead_right.index.tolist())
            # dead_idx = np.array(dead_idx)

            dead_sites = [siteMap[i] for i in dead_idx]
            dead_channels = [idx - 1 for idx in dead_sites]
            print([dead_idx[i] + 1 for i in range(len(dead_idx))])
            print(dead_channels)




