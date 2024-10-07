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
from timeit import default_timer as timer 
from histology_analysis_helpers import count_cells, save_results, define_roi, transform_roi, \
    get_cell_density, get_stack_stats, plot_example_slice, plot_bin_stats, calculate_fluo_intensities, count_cells_wrapper

Image.MAX_IMAGE_PIXELS = 500000000
logging.basicConfig(level=logging.DEBUG)

#%% Set parameters
if __name__ == "__main__": 
    parser = argparse.ArgumentParser()

    parser.add_argument("--um_per_pix", type=float, default=0.65, help="Pixel size in micrometers.")
    parser.add_argument("--probe_size", type=float, default=3500, help="Probe size in micrometers.")
    parser.add_argument("--hemi", type=str, default='left', help="Hemisphere.")
    parser.add_argument("--staining", type=str, default='GFAP', help="Type of staining.")
    parser.add_argument("--dr", type=int, default=25, help="Distance in micrometers.")
    parser.add_argument("--animal", type=str, default='rEO_05', help="Animal identifier.")
    parser.add_argument("--analysis", type=str, choices=['cell_counting','fluorescence'], default='cell_counting', help="Type of analysis (fluorescence/cell_counting).")
    parser.add_argument("--forceReload", type=int, choices=[0, 1], default=1, help="Force reload of data.")
    parser.add_argument("--use_multiprocessing", type=int, choices=[0, 1], default=1, help="Use multiprocessing.")
    parser.add_argument("--input_dir", type=str, default='H:/histology/new_scans/rEO_05/1x', help="Path to analysis input (where raw images are)." )
    parser.add_argument("--output_dir", type=str, default='H:/histology/analyzed_slices/rEO_05/multiprocessing/new_roi', help="Path to analysis output.")
    parser.add_argument("--example_slice", type=str, default='H:/histology/example_slices/rEO_05/rEO_05_s7_n2_color_left_GFAP.png', help="Path to example slice image.")
    parser.add_argument("--slices", type=str, nargs='+', default=['s6_n1','s7_n2','s7_n1'], help="3 brain slices around the ROI where the probe is seen.")
    parser.add_argument("--probe_line_begin", type=int, nargs=2, default=[1878, 8139], help="Coordinates of probe line beginning (two integers in px).")
    parser.add_argument("--probe_line_end", type=int, nargs=2, default=[2096, 2754], help="Coordinates of probe line end (two integers in px).")
    parser.add_argument("--crop_coords", type=int, nargs=8, default=[1219, 6095, 1167, 7275, 2652, 7325, 2705, 6144], help="Cropping coordinates (eight integers in px).")
    parser.add_argument("--num_images_per_slice", type=int, default=1, help="Number of images for each brain slice. 1 if all ROIs are on the same image / n for multiple images.")
    parser.add_argument("--num_chs_per_cell", type=int, default=1, help="Number of imaging channels used for cell counting. 2 if DAPI staining is available.")
    parser.add_argument("--staining_suppl", type=str, default='DAPI', help="Supplementary staining per channel e.g. DAPI.")
    parser.add_argument('--debug', action='store_true', help='Run in debug mode.')

    args = parser.parse_args()

    um_per_pix = args.um_per_pix
    probe_size = args.probe_size 
    hemi = args.hemi
    staining = args.staining
    dr = args.dr 
    animal = args.animal
    analysis = args.analysis  # fluorescence / cell_counting
    forceReload = args.forceReload
    use_multiprocessing = args.use_multiprocessing
    input_dir = args.input_dir
    output_dir = args.output_dir
    example_slice = args.example_slice
    slices = args.slices
    probe_line_begin = args.probe_line_begin
    probe_line_end = args.probe_line_end
    crop_coords = args.crop_coords
    num_images_per_slice = args.num_images_per_slice
    num_chs_per_cell = args.num_chs_per_cell
    staining_suppl= args.staining_suppl

    if args.debug:
        pdb.set_trace()

    if num_chs_per_cell != 1:
        assert(staining_suppl is not None)

    output_dir_staining = os.path.join(output_dir, staining, hemi, 'r'+str(dr), analysis)
    if not os.path.exists(output_dir_staining):
        os.makedirs(output_dir_staining)

    if num_chs_per_cell != 1 and staining != staining_suppl:
        output_dir_staining_suppl = os.path.join(output_dir, staining_suppl, hemi, 'r'+str(dr), analysis)
   
    # Define what images to load based on chosen staining(s) to analyze
    image_dir = os.path.join(input_dir, staining)
    all_files = [os.path.join(image_dir, filename) for filename in (os.listdir(image_dir))]
    image_files = [file for file in all_files if animal in os.path.splitext(os.path.basename(file))[0] and \
                   any(s in os.path.basename(file) for s in slices)] 
    print(image_files)

    # Define probe trajectory
    probe_line_points = int(np.floor(probe_size / um_per_pix))  # 3.5 mm 
    probe_line_x = np.linspace(probe_line_begin[0], probe_line_end[0], probe_line_points, dtype=np.single)
    probe_line_y = np.linspace(probe_line_begin[1], probe_line_end[1], probe_line_points, dtype=np.single)

    # Define area around the probe (~1 mm / 500 um on each side)
    crop_dims = ((crop_coords[4]-crop_coords[2]),(crop_coords[3]-crop_coords[1])) # x,y
    radii = np.linspace(500, 0, int((500+dr)/dr))

    line_begin = [crop_coords[2]+(crop_coords[4]-crop_coords[2])/2, crop_coords[3]+(crop_coords[5]-crop_coords[3])/2]
    line_end = [crop_coords[0]+(crop_coords[6]-crop_coords[0])/2, crop_coords[1]+(crop_coords[7]-crop_coords[1])/2]
    line_points = int(np.rint(np.sqrt((line_end[0]-line_begin[0])**2 + (line_end[1]-line_begin[1])**2)))

    line_x = np.linspace(line_begin[0], line_end[0], line_points, dtype=np.single)
    line_y = np.linspace(line_begin[1], line_end[1], line_points, dtype=np.single)
    
    
    #%% Analyze the images
    for f, file in enumerate(image_files):
        print(file)
        filename = os.path.splitext(os.path.basename(file))[0]
        filename_parts = filename.split('_')[:-1]
        slice_ID = '_'.join(filename_parts)
        
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

        # Define a ROI around the probe (1 mm)
        radius_images = define_roi(image, imarray, line_x, line_y, radii, um_per_pix)
        
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
                    num_cells, cells_pos = count_cells_wrapper(image, radius_images, radii, staining, num_chs_per_cell=num_chs_per_cell, cells_pos_suppl=cells_pos_suppl, staining_suppl=staining_suppl)
                    print("with parallel processing:", timer()-start) 
                else:
                    num_cells, cells_pos = count_cells(image, radius_images, radii, staining, output_file, num_chs_per_cell=num_chs_per_cell, cells_pos_suppl=cells_pos_suppl, staining_suppl=staining_suppl, plot=False)
                    print("with serial processing:", timer()-start)

            else:
                # Count the number of cells and get their coordinates 
                print('Counting cells...')
                start = timer() 
                if use_multiprocessing:
                    num_cells, cells_pos = count_cells_wrapper(image, radius_images, radii, staining, num_chs_per_cell=num_chs_per_cell)
                    print("with parallel processing:", timer()-start) 
                else:
                    num_cells, cells_pos = count_cells(image, radius_images, radii, staining, output_file, num_chs_per_cell=num_chs_per_cell, plot=False)
                    print("with serial processing:", timer()-start)

            # Get the density of cells for each slice as % of the first column in the slice
            perc_cell_density = get_cell_density(radius_images, radii, num_cells, um_per_pix)
            
            # Crop the ROI and superimpose detected cells
            transform_roi(imarray, line_x, line_y, radii, crop_coords, crop_dims, output_file, cells_pos=cells_pos, plot=True)
            # output_file =  output_file + '_testFilt'
            # transform_roi(imarray, line_x, line_y, radii, crop_coords, crop_dims, output_file, cells_pos=cells_pos_filt, plot=True)
            # Save results in csv
            save_results(output_file, slice_ID, radii, analysis, num_cells=num_cells, cells_pos=cells_pos, perc_cell_density=perc_cell_density, fluo=None)

        elif analysis == 'fluorescence':
            print('Measuring fluorescence...')
            fluo = calculate_fluo_intensities(image, radius_images, radii)

            save_results(output_file, slice_ID, radii, analysis, num_cells=None, cells_pos=None, perc_cell_density=None, fluo=fluo)
            

    #%% Analyze results of all images of a certain channel
    mean_cell_density, std_cell_density = get_stack_stats(output_dir_staining, analysis)

    # Plot example slice with mean +/- std cell density or fluorescence intensity
    filename = os.path.splitext(os.path.basename(example_slice))[0]
    filename_parts = filename.split('_')[:-1]
    slice_ID = '_'.join(filename_parts)
    if analysis == 'fluorescence':
        output_file = os.path.join(output_dir_staining, staining + '_' + slice_ID + '_fluorescence_' + hemi)
    elif analysis == 'cell_counting':
        output_file = os.path.join(output_dir_staining, staining + '_' + slice_ID + '_cell_density_' + hemi)
        
    plot_example_slice(example_slice, mean_cell_density, std_cell_density, line_x, line_y, radii, crop_coords, crop_dims, output_file, um_per_pix, dr)    

    # Plot iterative change in cell density / fluorescence only
    if analysis == 'fluorescence':
        output_file = os.path.join(output_dir_staining, staining + '_plot_fluorescence_' + hemi)
    elif analysis == 'cell_counting':
        output_file = os.path.join(output_dir_staining, staining + '_plot_cell_density_' + hemi)

    plot_bin_stats(mean_cell_density, std_cell_density, radii, crop_dims, um_per_pix, dr, output_file)