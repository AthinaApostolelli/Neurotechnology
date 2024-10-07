"""
@author: Athina Apostolelli - July 2024 


INPUT PARAMETERS


OUTPUT
    - csv file 
        

    - barplot of cell density or fluorescence intensity, comparing the ROIs around and away from the probe and the two hemispheres
"""

from matplotlib import pyplot as plt
from scipy.stats import stats
import os, glob, csv
import numpy as np
from histology_analysis_helpers import convert_pvalue_to_asterisks, get_stack_stats_pyr, plot_on_off_bars

analysis = 'cell_counting'
animal = 'rEO_05'
staining = 'NISSL'

output_dir = 'H:/histology/analyzed_slices/' + animal + '/roi_pyr'
output_dir_left = os.path.join(output_dir, staining, 'left', analysis)
output_dir_right = os.path.join(output_dir, staining, 'right', analysis)

# Load cell densities - left hemi
csv_files_L = glob.glob(os.path.join(output_dir_left, '*cell_counts.csv'))

cell_density_slice_on_L = [[] for _ in range(len(csv_files_L))]
cell_density_slice_off_L = [[] for _ in range(len(csv_files_L))]
for i in range(len(csv_files_L)):
    results_file = csv_files_L[i]
    with open(results_file, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if analysis == 'fluorescence':
                cell_density_slice_on_L[i].append(row['fluorescence_on'])
                cell_density_slice_off_L[i].append(row['fluorescence_off'])
            elif analysis == 'cell_counting':
                cell_density_slice_on_L[i].append(row['cell_density_on'])
                cell_density_slice_off_L[i].append(row['cell_density_off'])

cell_density_slice_on_L = np.array(cell_density_slice_on_L, dtype=np.float64) 
cell_density_slice_off_L = np.array(cell_density_slice_off_L, dtype=np.float64)

mean_cell_density_on_L = np.mean(cell_density_slice_on_L, axis=0)
mean_cell_density_off_L = np.mean(cell_density_slice_off_L, axis=0)
std_cell_density_on_L = np.std(cell_density_slice_on_L, axis=0)
std_cell_density_off_L = np.std(cell_density_slice_off_L, axis=0)

# Load cell densities - right hemi
csv_files_R = glob.glob(os.path.join(output_dir_right, '*cell_counts.csv'))

cell_density_slice_on_R = [[] for _ in range(len(csv_files_R))]
cell_density_slice_off_R = [[] for _ in range(len(csv_files_R))]
for i in range(len(csv_files_R)):
    results_file = csv_files_R[i]
    with open(results_file, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if analysis == 'fluorescence':
                cell_density_slice_on_R[i].append(row['fluorescence_on'])
                cell_density_slice_off_R[i].append(row['fluorescence_off'])
            elif analysis == 'cell_counting':
                cell_density_slice_on_R[i].append(row['cell_density_on'])
                cell_density_slice_off_R[i].append(row['cell_density_off'])

cell_density_slice_on_R = np.array(cell_density_slice_on_R, dtype=np.float64) 
cell_density_slice_off_R = np.array(cell_density_slice_off_R, dtype=np.float64)

mean_cell_density_on_R = np.mean(cell_density_slice_on_R, axis=0)
mean_cell_density_off_R = np.mean(cell_density_slice_off_R, axis=0)
std_cell_density_on_R = np.std(cell_density_slice_on_R, axis=0)
std_cell_density_off_R = np.std(cell_density_slice_off_R, axis=0)

# Perform statistics (Kruskal-Wallis test)
data = [cell_density_slice_on_L, cell_density_slice_off_L, cell_density_slice_on_R, cell_density_slice_off_R]
group1 = [1,1,2,2] # compare left-right
group2 = [1,2,1,2] # compare on-off (around vs away from probe)
anova, pvalue = stats.kruskal(cell_density_slice_on_L, cell_density_slice_off_L, cell_density_slice_on_R, cell_density_slice_off_R)

# Plot bar plots with significance indicated 

        
# def plot_on_off_bars(mean_cell_density_on, mean_cell_density_off, std_cell_density_on, std_cell_density_off, pvalue, output_file):
#     '''
#     Plot the cell density or fluorescence intensity, comparing the ROIs around and away from the probe
#     '''
#     fig, ax = plt.subplots(figsize=(5,5))
#     x_values = [0,0.15]
#     ax.bar(x_values, [mean_cell_density_on[0], mean_cell_density_off[0]], yerr=[std_cell_density_on[0], std_cell_density_off[0]], \
#            width=0.1, error_kw=dict(elinewidth=2, capsize=5))

#     # Plot significance 
#     ax_y0, ax_y1 = plt.gca().get_ylim()
#     dh = (ax_y1 - ax_y0)/5
#     barh = (ax_y1 - ax_y0)/30
#     barx = [x_values[0],x_values[0],x_values[1],x_values[1]]
#     y = max([mean_cell_density_on[0], mean_cell_density_off[0]]) + dh
#     bary = [y, y+barh, y+barh, y]
#     mid = ((x_values[0]+x_values[1])/2, y + barh*1.2)

#     plt.plot(barx, bary, c='black')
#     plt.text(*mid, convert_pvalue_to_asterisks(pvalue), fontsize=14)
#     ax.set_xticks(x_values)
#     ax.set_xticklabels(["Around\nprobe","Away from\nprobe"], fontsize=14)
#     # ax.ticklabel_format(axis='y', style='sci', scilimits=(-4,-4), useMathText=True)
#     ax.ticklabel_format(axis='y', style='sci', useMathText=True)

#     plt.ylabel('Cell density (100 um^2)', fontsize=14)
#     plt.rcParams['svg.fonttype'] = 'none'
#     plt.rcParams['axes.spines.right'] = False
#     plt.rcParams['axes.spines.top'] = False
#     plt.savefig(output_file + '.png')
#     plt.savefig(output_file + '.svg', format='svg')
#     plt.close()    

    

