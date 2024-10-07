#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Athina Apostolelli - April 2024 
Adapted from Tansel Baran Yasar 

Use python version > 3.10 to get the most updated packages

INPUT PARAMETERS
   - um_per_pix              pixel size in um
   - probe_size              length of probe in um
   - hemi                    hemisphere to analyse
   - staining                type of staining to analyse
   - dr                      size of vertical bin around probe in um  
   - animal                  animal ID (relevant for file naming)
   - analysis                whether to count cells ('cell_counting') or measure fluorescence intensity ('fluorescence')
   - forceReload             'true' if the results should be overwritten 
   - use_multiprocessing     'true' to use pool of processes with shared memory 
   - output_dir              general output directory
   - example_slice           image to use to plot superimposed cell density or fluorescence intensity
   - probe_line_begin        coordinates of deepest probe point - needs to be consistent across images
   - probe_line_end          coordinates of most superficial probe point - needs to be consistent across images
   - crop_coords             coordinates of ROI - needs to be consistent across images

OUTPUT
    - csv file 
        if 'cell_counting' analysis
        - cell density for each bin as % change from reference bin (previous bin) 
        - coordinates of each cell detected      
        if 'fluorescence' analysis
        - fluorescence intensity for each bin as % change from reference bin (previous bin)

    - image of ROI with marked cells for each slice ('cell_counting' analysis)
    - image of example ROI with superimposed plot of cell density or fluorescence intensity
"""

#%%
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import os, logging, argparse, errno, pdb, ast
import pandas as pd
from matplotlib.patches import Polygon
import cv2
from timeit import default_timer as timer 
from histology_analysis_helpers import count_cells, save_results, define_roi, transform_roi, transform_roi_pyr, count_cells_pyr, \
    get_cell_density, get_stack_stats, plot_example_slice, plot_bin_stats, calculate_fluo_intensities, count_cells_wrapper, \
    get_cell_density_pyr, save_results_pyr, get_stack_stats_pyr, plot_on_off_bars

Image.MAX_IMAGE_PIXELS = 500_000_000
# logging.basicConfig(level=logging.DEBUG)

def parse_coords(coords_str):
    coords_list = coords_str.split(',')
    coords_array = [list(map(int, coords.split())) for coords in coords_list]
    return coords_array

#%% Set parameters
if __name__ == "__main__": 
    # parser = argparse.ArgumentParser()

    # parser.add_argument("--um_per_pix", type=float, default=0.65, required=True, help="Pixel size in micrometers.")
    # parser.add_argument("--probe_size", type=float, default=3500, required=True, help="Probe size in micrometers.")
    # parser.add_argument("--hemi", type=str, default='left', required=True, help="Hemisphere.")
    # parser.add_argument("--staining", type=str, default='IBA', required=True, help="Type of staining.")
    # parser.add_argument("--dr", type=int, default=25, required=True, help="Distance in micrometers.")
    # parser.add_argument("--animal", type=str, default='rEO_05', required=True, help="Animal identifier.")
    # parser.add_argument("--analysis", type=str, choices=['cell_counting','fluorescence'], default='cell_counting', required=True, help="Type of analysis (fluorescence/cell_counting).")
    # parser.add_argument("--forceReload", type=int, choices=[0, 1], default=1, required=True, help="Force reload of data.")
    # parser.add_argument("--use_multiprocessing", type=int, choices=[0, 1], default=0, required=True, help="Use multiprocessing.")
    # parser.add_argument("--input_dir", type=str, default='H:/histology/new_scans/rEO_05/1x', required=True, help="Path to analysis input (where raw images are)." )
    # parser.add_argument("--output_dir", type=str, default='H:/histology/analyzed_slices/rEO_05/roi_pyr', required=True, help="Path to analysis output.")
    # parser.add_argument("--example_slice", type=str, default='H:/histology/example_slices/rEO_05/rEO_05_s5_n1_color_left_IBA.png', required=True, help="Path to example slice image.")
    # parser.add_argument("--slices", type=str, nargs='+', default=['s6_n2','s5_n1','s5_n2'], help="3 brain slices around the ROI where the probe is seen.")
    # parser.add_argument("--probe_line_begin", type=int, nargs=2, default=[1878, 8139], help="Coordinates of probe line beginning (two integers in px).")
    # parser.add_argument("--probe_line_end", type=int, nargs=2, default=[2096, 2754], help="Coordinates of probe line end (two integers in px).")
    # parser.add_argument("--crop_coords_on", type=str, default=[1624, 3616, 1849, 3919, 2459, 3453, 2234, 3150, 1401, 3726, 1658, 4002, 2212, 3470, 1955, 3194, 1195, 3577, 1391, 3890, 2041, 3492, 1846, 3169], required=True, help="Cropping coordinates (eight integers in px).")
    # parser.add_argument("--crop_coords_off", type=str, default=[2890, 3062, 2801, 3429, 3550, 3598, 3639, 3231, 2890, 2992, 2801, 3359, 3550, 3528, 3639, 3161, 2716, 2754, 2627, 3120, 3376, 3290, 3465, 2923], required=True, help="Cropping coordinates (eight integers in px).")
    # parser.add_argument("--num_images_per_slice", type=int, default=1, help="Number of images for each brain slice. 1 if all ROIs are on the same image / n for multiple images.")
    # parser.add_argument("--num_chs_per_cell", type=int, default=1, help="Number of imaging channels used for cell counting. 2 if DAPI staining is available.")
    # parser.add_argument("--staining_suppl", type=str, default='DAPI', help="Supplementary staining per channel e.g. DAPI.")
    # parser.add_argument('--debug', action='store_true', help='Run in debug mode.')

    # args = parser.parse_args()

    # um_per_pix = args.um_per_pix
    # probe_size = args.probe_size 
    # hemi = args.hemi
    # staining = args.staining
    # dr = args.dr 
    # animal = args.animal
    # analysis = args.analysis  # fluorescence / cell_counting
    # forceReload = args.forceReload
    # use_multiprocessing = args.use_multiprocessing
    # input_dir = args.input_dir
    # output_dir = args.output_dir
    # example_slice = args.example_slice
    # slices = args.slices
    # probe_line_begin = args.probe_line_begin
    # probe_line_end = args.probe_line_end
    # num_images_per_slice = args.num_images_per_slice
    # num_chs_per_cell = args.num_chs_per_cell
    # staining_suppl = args.staining_suppl

    um_per_pix = 0.65
    probe_size = 3500
    hemi = 'right'
    staining = 'NISSL'
    dr = 25
    animal = 'rEO_05'
    analysis = 'cell_counting'  # fluorescence / cell_counting
    forceReload = 1
    use_multiprocessing = 0
    if staining == 'IBA':
        input_dir = 'H:/histology/new_scans/' + animal + '/1x_correctBG'
    elif staining == 'NISSL':
        input_dir = 'H:/histology/new_scans/' + animal + '/1x'
    output_dir = 'H:/histology/analyzed_slices/' + animal + '/roi_pyr'
    example_slice = 's5_n1'
    slices = 's5_n2', 's5_n1', 's6_n2'
    # probe_line_begin = [1878, 8139]
    # probe_line_end = [2096, 2754]
    num_images_per_slice = 1
    num_chs_per_cell = 1
    staining_suppl = None
    crop_coords_on_slices = [8270, 2886, 8174, 3251, 8919, 3435, 9015, 3070, 8216, 2920, 8120, 3285, 8866, 3469, 8962, 3104, 8216, 2832, 8120, 3197, 8866, 3381, 8962, 3016]
    crop_coords_off_slices = [6905, 3580, 7133, 3881, 7738, 3408, 7510, 3107, 6905, 3580, 7133, 3881, 7738, 3408, 7510, 3107, 6799, 3504, 7027, 3805, 7632, 3332, 7404, 3031]

    # crop_coords_on_slices = list(map(int, args.crop_coords_on.split()))
    # crop_coords_off_slices = list(map(int, args.crop_coords_off.split()))

    crop_coords_on_slices = [crop_coords_on_slices[i:i + 8] for i in range(0, len(crop_coords_on_slices), 8)]
    crop_coords_off_slices = [crop_coords_off_slices[i:i + 8] for i in range(0, len(crop_coords_off_slices), 8)]  

    # if args.debug:
    #     pdb.set_trace()

    if num_chs_per_cell != 1:
        assert(staining_suppl is not None)    

    output_dir_staining = os.path.join(output_dir, staining, hemi, analysis)
    if not os.path.exists(output_dir_staining):
        os.makedirs(output_dir_staining)

    if num_chs_per_cell != 1 and staining != staining_suppl:
        output_dir_staining_suppl = os.path.join(output_dir, staining_suppl, hemi, analysis)
   
    # Define what images to load based on chosen staining(s) to analyze
    image_dir = os.path.join(input_dir, staining)
    all_files = [os.path.join(image_dir, filename) for filename in (os.listdir(image_dir))]
    image_files = [file for file in all_files if animal in os.path.splitext(os.path.basename(file))[0] and \
                   any(s in os.path.basename(file) for s in slices)] 
    slice_order = {slice_name: index for index, slice_name in enumerate(slices)} # sort by slice order
    image_files = sorted(image_files, key=lambda file: slice_order[next(s for s in slices if s in os.path.basename(file))])


    #%% Analyze the images
    for f, file in enumerate(image_files):
        print(file)
        filename = os.path.splitext(os.path.basename(file))[0]
        filename_parts = filename.split('_')[:-1]
        slice_ID = '_'.join(filename_parts)

        # Define ROIs
        crop_coords_on = np.array(crop_coords_on_slices[f], dtype=np.int32)
        crop_coords_off = np.array(crop_coords_off_slices[f], dtype=np.int32)

        # Define output file
        if not os.path.exists(output_dir_staining):
            os.makedirs(output_dir_staining)
        output_file = os.path.join(output_dir_staining, staining + '_' + slice_ID + '_' + hemi + '_correctBG') # background subtracted images
        
        # Load image
        image = Image.open(file)
        imarray = np.asarray(image)

        # plt.figure(1)
        # plt.imshow(image)
        # plt.plot(probe_line_x,probe_line_y,color='white')
        # plt.show()
        
        # plt.figure(2)
        # plt.imshow(image)
        # plt.plot(line_x,line_y,color='white')
        # plt.show()

        # Check if the image has already been analyzed
        if analysis == 'cell_counting':
            results_file = output_file + '_cell_counts.csv'
        elif analysis == 'fluorescence':
            results_file = output_file + '_fluorescence.csv'

        if os.path.exists(results_file) and not forceReload:
            print('This image has already been analyzed.')
            continue

        # elif os.path.exists(results_file) and forceReload:
        #     # Read results from csv file
        #     df = pd.read_csv(results_file)
        #     num_cells = df['num_cells'].apply(lambda x: ast.literal_eval(x)) # convert from str

        #     # Calculate new cell_density values and overwrite csv file
        #     radius_images = define_roi(image, imarray, line_x, line_y, radii, um_per_pix)
        #     new_cell_density = get_cell_density(radius_images, radii, num_cells, um_per_pix)
        #     df['cell_density'] = {k: round(v, 2) for k, v in new_cell_density.items()}
        #     df.to_csv(results_file, index=False)

        #     print("Cell density recalculated and CSV file updated successfully.") 
        #     continue

        # Define a ROI around the probe (1 mm) and in a different area 
        roi_on_bin_image = np.zeros((imarray.shape[0],imarray.shape[1]), dtype='int8')
        roi_on_bin_image = cv2.fillPoly(roi_on_bin_image, crop_coords_on.reshape((1, 4, 2)), 1)

        roi_off_bin_image = np.zeros((imarray.shape[0],imarray.shape[1]), dtype='int8')
        roi_off_bin_image = cv2.fillPoly(roi_off_bin_image, crop_coords_off.reshape((1, 4, 2)), 1)
        
        # (Optional) display the binary image
        # plt.imshow(roi_on_bin_image, cmap='gray')
        # plt.title('Binary Image with ROI')
        # plt.show()

        if analysis == 'cell_counting':
            if num_chs_per_cell != 1 and staining != staining_suppl:

                # Check if supplementary staining has been analyzed and load data, if appropriate
                results_file_suppl = os.path.join(output_dir_staining_suppl, staining_suppl + '_' + slice_ID + '_' + hemi + '_correctBG_cell_counts.csv')

                if os.path.exists(results_file_suppl):
                    df = pd.read_csv(results_file_suppl)
                    df['cells_pos'] = df['cells_pos'].apply(lambda x: ast.literal_eval(x)) # convert from str
                    cells_pos_suppl = sum(df['cells_pos'], [])
                    # num_cells_suppl = df['num_cells'].tolist()
                    print('Loaded supplementary staining cell counts and positions.')
                else:
                    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), results_file_suppl)
                
                # Count the number of cells and get their coordinates 
                print('Counting cells using supplementary staining...')
                start = timer() 
                if use_multiprocessing:
                    raise Exception("Multiprocessing has not been implemented for pyramidal layer analysis yet.")
                else:
                    num_cells_on, cells_pos_on = count_cells_pyr(image, roi_on_bin_image, staining, output_file, num_chs_per_cell=num_chs_per_cell, cells_pos_suppl=cells_pos_suppl, staining_suppl=staining_suppl, plot=False)
                    num_cells_off, cells_pos_off = count_cells_pyr(image, roi_off_bin_image, staining, output_file, num_chs_per_cell=num_chs_per_cell, cells_pos_suppl=cells_pos_suppl, staining_suppl=staining_suppl, plot=False)
                    print("with serial processing:", timer()-start)

            else:
                # Count the number of cells and get their coordinates 
                print('Counting cells...')
                start = timer() 
                if use_multiprocessing:
                    raise Exception("Multiprocessing has not been implemented for pyramidal layer analysis yet.")
                else:
                    num_cells_on, cells_pos_on = count_cells_pyr(image, roi_on_bin_image, staining, output_file, num_chs_per_cell=num_chs_per_cell, cells_pos_suppl=None, staining_suppl=None, plot=True)
                    num_cells_off, cells_pos_off = count_cells_pyr(image, roi_off_bin_image, staining, output_file, num_chs_per_cell=num_chs_per_cell, cells_pos_suppl=None, staining_suppl=None, plot=True)
                    print("with serial processing:", timer()-start)

            # Get the density of cells for each slice as % of the first column in the slice
            cell_density_on, cell_density_off = get_cell_density_pyr(roi_on_bin_image, roi_off_bin_image, num_cells_on, num_cells_off, um_per_pix)
            
            # Crop the ROI and superimpose detected cells
            transform_roi_pyr(imarray, crop_coords_on, output_file + '_ON', cells_pos=cells_pos_on, plot=True)
            transform_roi_pyr(imarray, crop_coords_off, output_file + '_OFF', cells_pos=cells_pos_off, plot=True)
            
            # Save results in csv
            save_results_pyr(output_file, slice_ID, analysis, num_cells_on=num_cells_on, num_cells_off=num_cells_off, \
                        cells_pos_on=cells_pos_on, cells_pos_off=cells_pos_off, cell_density_on=cell_density_on, \
                        cell_density_off=cell_density_off, fluo_on=None, fluo_off=None)

        elif analysis == 'fluorescence':
            raise Exception("Fluorescence analysis has not been implemented for pyramidal layer analysis yet.")
            # print('Measuring fluorescence...')
            # fluo = calculate_fluo_intensities(image, radius_images, radii)

            # save_results(output_file, slice_ID, radii, analysis, num_cells=None, cells_pos=None, perc_cell_density=None, fluo=fluo)
            

    #%% Analyze results of all images of a certain channel
    mean_cell_density_on, mean_cell_density_off, std_cell_density_on, std_cell_density_off, t_test, pvalue = get_stack_stats_pyr(output_dir_staining, analysis)

    # Plot bars of mean cell density / fluorescence around or away from probe
    if analysis == 'fluorescence':
        output_file = os.path.join(output_dir_staining, staining + '_plot_fluorescence_' + hemi)
    elif analysis == 'cell_counting':
        output_file = os.path.join(output_dir_staining, staining + '_plot_cell_density_' + hemi)

    plot_on_off_bars(mean_cell_density_on, mean_cell_density_off, std_cell_density_on, std_cell_density_off, pvalue, output_file)