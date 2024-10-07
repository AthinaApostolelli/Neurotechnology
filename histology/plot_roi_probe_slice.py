#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Athina Apostolelli - June 2024 
Adapted from Tansel Baran Yasar

Use python version > 3.10 to get the most updated packages
The code runs well in the 'histology' conda environment
"""

from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import os
from matplotlib.patches import Polygon

Image.MAX_IMAGE_PIXELS = 500000000

output_dir = 'H:/histology/example_slices'
um_per_pix = 0.325 #1.625#
staining = 'GFAP'
dr = 25
animal = 'rEO_05'

file = 'H:/histology/example_slices/rEO_05_s7_n1_color_' + staining + '.png'
filename = os.path.splitext(os.path.basename(file))[0]
filename_parts = filename.split('_')[:-1]
slice_ID = '_'.join(filename_parts)
output_file = os.path.join(output_dir, staining + '_' + slice_ID + '_probe_roi')

# RIGHT HEMI
# Define probe trajectories
probe_line_begin1 = [4336,15847] #[3393, 3346]#
probe_line_end1 = [4899,5193] #[3507, 1192]#
probe_line_points1 = 10670 #int(np.floor(3500 / um_per_pix))  # 3.5 mm 

probe_line_x1 = np.linspace(probe_line_begin1[0], probe_line_end1[0], probe_line_points1, dtype=np.single)
probe_line_y1 = np.linspace(probe_line_begin1[1], probe_line_end1[1], probe_line_points1, dtype=np.single)

# Define short trajectory to look around
# Define 8-tuple with x,y coordinates of top-left, bottom-left, bottom-right and top-right
crop_coords1 = [3264,6077,2909,12649,5971,12814,6326,6242] #[3176,1370,3104,2700,3724,2733,3795,1404]#
crop_dims1 = ((crop_coords1[4]-crop_coords1[2]),(crop_coords1[3]-crop_coords1[1])) # x,y

line_begin1 = [crop_coords1[2]+(crop_coords1[4]-crop_coords1[2])/2, crop_coords1[3]+(crop_coords1[5]-crop_coords1[3])/2]
line_end1 = [crop_coords1[0]+(crop_coords1[6]-crop_coords1[0])/2, crop_coords1[1]+(crop_coords1[7]-crop_coords1[1])/2]
line_points1 = int(np.rint(np.sqrt((line_end1[0]-line_begin1[0])**2 + (line_end1[1]-line_begin1[1])**2)))

line_x1 = np.linspace(line_begin1[0], line_end1[0], line_points1, dtype=np.single)
line_y1 = np.linspace(line_begin1[1], line_end1[1], line_points1, dtype=np.single)

# LEFT HEMI
probe_line_begin2 = [18227,13692] #[6269, 2834]
probe_line_end2 = [17777,3037] #[6178, 680]
probe_line_points2 = 10670  # 3.5 mm = 2154 pixels/points

probe_line_x2 = np.linspace(probe_line_begin2[0], probe_line_end2[0], probe_line_points2, dtype=np.single)
probe_line_y2 = np.linspace(probe_line_begin2[1], probe_line_end2[1], probe_line_points2, dtype=np.single)

# Define short trajectory to look around
# Define 8-tuple with x,y coordinates of top-left, bottom-left, bottom-right and top-right
crop_coords2 = [16345,5135,16594,11711,19659,11595,19409,5018] #[5888,1104,5939,2434,6558,2410,6508,1081]
crop_dims2 = ((crop_coords2[4]-crop_coords2[2]),(crop_coords2[3]-crop_coords2[1])) # x,y

line_begin2 = [crop_coords2[2]+(crop_coords2[4]-crop_coords2[2])/2, crop_coords2[3]+(crop_coords2[5]-crop_coords2[3])/2]
line_end2 = [crop_coords2[0]+(crop_coords2[6]-crop_coords2[0])/2, crop_coords2[1]+(crop_coords2[7]-crop_coords2[1])/2]
line_points2 = int(np.rint(np.sqrt((line_end2[0]-line_begin2[0])**2 + (line_end2[1]-line_begin2[1])**2)))

line_x2 = np.linspace(line_begin2[0], line_end2[0], line_points1, dtype=np.single)
line_y2 = np.linspace(line_begin2[1], line_end2[1], line_points1, dtype=np.single)


# Define area around the probe (~1 mm / 500 um on each side)
radii = np.linspace(500, 0, int((500+dr)/dr))

# Define polygons
tl1 = (crop_coords1[0], crop_coords1[1])
bl1 = (crop_coords1[2], crop_coords1[3])
br1 = (crop_coords1[4], crop_coords1[5])
tr1 = (crop_coords1[6], crop_coords1[7])

tl2 = (crop_coords2[0], crop_coords2[1])
bl2 = (crop_coords2[2], crop_coords2[3])
br2 = (crop_coords2[4], crop_coords2[5])
tr2 = (crop_coords2[6], crop_coords2[7])

rect_coords1 = [tl1, bl1, br1, tr1]
rect_coords2 = [tl2, bl2, br2, tr2]
polygon1 = Polygon(rect_coords1, closed=True, edgecolor='w', facecolor='none', linewidth=2)
polygon2 = Polygon(rect_coords2, closed=True, edgecolor='w', facecolor='none', linewidth=2)


# Load image
image = Image.open(file)
image = image.convert('RGB')
# imarray = np.asarray(image)

plt.figure(1)
plt.imshow(image)
plt.plot(probe_line_x1,probe_line_y1,color='white')
plt.plot(probe_line_x2,probe_line_y2,color='white')
plt.gca().add_patch(polygon1)
plt.gca().add_patch(polygon2)
plt.gca().set_axis_off()
plt.rcParams['svg.fonttype'] = 'none'
plt.savefig(output_file + '.png')
plt.savefig(output_file + '.svg', format='svg')
plt.close()


