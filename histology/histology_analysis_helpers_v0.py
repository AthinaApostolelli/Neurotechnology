#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Athina Apostolelli - April 2024 
Adapted from Tansel Baran Yasar 

Use python version > 3.10 to get the most updated packages
"""
#%%
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.signal import convolve 
from scipy import ndimage as ndi
from skimage import measure
from skimage.morphology import disk, closing, remove_small_objects, remove_small_holes
from skimage.segmentation import watershed
from skimage.color import rgb2gray
from skimage.filters import rank, gaussian
from skimage.util import img_as_ubyte
import os, csv, sys, glob, cv2
import SimpleITK as sitk
from scipy.spatial.distance import pdist, squareform
import plotly.graph_objects as go
from numba import jit, cuda
import cupy as cp
import logging
from multiprocessing import Pool, shared_memory, Manager


def calculate_fluo_intensities(image, radius_images, radii):
    '''
    Calculate the change in fluorescence intensity as % from reference bin
    '''
    image = image.convert('L')
    image = np.asarray(image)
    image = img_as_ubyte(image)

    fluo = {} 
    for r in range(len(radii)*2-2):
        roi = np.nonzero(image * radius_images[r])
        fluo[r] = np.mean(image[roi])

    perc_fluo = {}
    for r in range(len(radii)*2-2):
        if r == 0:
            perc_fluo[r] = float((fluo[r] - fluo[r]) / fluo[r] * 100)
            # perc_fluo[r] = (fluo[r] / fluo[r]) * 100
        else:
            # perc_change = (fluo[r] - fluo[r-1]) / fluo[r-1] 
            # perc_fluo[r] = (perc_fluo[r-1] + perc_change * perc_fluo[r-1]) 
            perc_fluo[r] = float((fluo[r] - fluo[r-1]) / fluo[r-1] * 100)

    return perc_fluo 


def calculate_fluo_intensities_v0(imarray, radius_images, radii):
    fluo = {} 
    for r in range(1,len(radii)):
        non_zero_points = np.nonzero(imarray*radius_images[r])
        fluo[r] = imarray[non_zero_points[1],non_zero_points[0]]
    return fluo 


def count_cells(image, radius_images, radii, staining, output_file, num_chs_per_cell=1, cells_pos_suppl=None, staining_suppl=None, plot=False):
    '''
    Count the number of cells in each bin (radius). Double-counted cells in two bins are removed from one of the two bins.

    Note that the parameters here e.g., threshold for markers detection, 
    area of cells etc. have been empirically determined for each staining type. 
    '''
    num_cells = {}
    cells_pos = {}
    areas = {}
    image = image.convert('RGB')
    # image = rgb2gray(image)
    image = np.asarray(image)
    image = img_as_ubyte(image)
    image = biascorrect(image)
    image = gaussian(image, sigma=3)
    image = img_as_ubyte(image)

    # Supplementary cells
    # centroids2 = [(int(y), int(x)) for y, x in cells_pos_suppl if 0 <= int(y) <image.shape[0] and 0 <= int(x) < image.shape[1]]
    # cells2 = [(y, x) for y, x in cells_pos_suppl if 0 <= y <image.shape[0] and 0 <= x < image.shape[1]]
    # cells_pos = {r: [] for r in range(len(radii)*2-2)}
    # num_cells = {r: [] for r in range(len(radii)*2-2)}

    # for r in range(len(radii)*2-2):
    #     # radius_images_concat = np.sum(radius_images, axis=0)
    #     # roi = np.nonzero(image * radius_images_concat)
    #     roi = np.nonzero(image * radius_images[r])
    #     roi_image = np.zeros((image.shape[0],image.shape[1]), dtype='uint8')
    #     roi_image[roi[0],roi[1]] = image[roi]

    #     mask = np.full_like(image, False, dtype=bool)
    #     mask[roi] = True

    #     binary_image = rank.gradient(roi_image, disk(2), mask=mask) > 20

    #     keep_indices = []
    #     for (idx, c2), cell2 in enumerate(zip(centroids2, cells2)):
    #         if binary_image[c2[0], c2[1]] == 1:
    #             keep_indices.append(idx)
    #             cells_pos[r].append(c2)

    #     cells_pos[r] = [cell for i, cell in enumerate(cells_pos[r])]
    #     num_cells[r] = len(cells_pos[r])

    # return cells_pos, num_cells
    # mask[centroids2[:,0], centroids2[:,1]] = True
        
    for r in range(len(radii)*2-2):
        radius_images_concat = np.sum(radius_images, axis=0)
        roi = np.nonzero(image * radius_images_concat)
        # roi = np.nonzero(image * radius_images[r])
        roi_image = np.zeros((image.shape[0],image.shape[1]), dtype='uint8')
        roi_image[roi[0],roi[1]] = image[roi]

        mask = np.full_like(image, False, dtype=bool)
        mask[roi] = True

        # Smooth the image to remove high frequency objects (e.g. dendrites)
        # roi_image_smooth = gaussian(roi_image, sigma=3)
        # roi_image = img_as_ubyte(roi_image_smooth)

        if staining == 'IBA':
            threshold_distance = 40
            binary_image = rank.gradient(roi_image, disk(2), mask=mask) > 15
            
            closed_image = closing(binary_image, disk(5)) # dilation followed by erosion
            closed_image = remove_small_holes(closed_image, 300, connectivity=2)
            closed_image = remove_small_objects(closed_image, 500, connectivity=2)
            closed_image = closed_image.astype(np.uint8)

            kernel = np.ones((3,3),np.uint8)
            dilation = cv2.dilate(closed_image, kernel, iterations = 5)
            
            # Continuous region (background): 1 = background
            #markers = rank.gradient(dilation, disk(7), mask=mask) == 0#17# 35
            #markers = ndi.label(markers)[0]

            # Process the watershed
            #labels = watershed(dilation, markers, connectivity=2)

            # Find cell coords
            labels = ndi.label(dilation)[0]
            centroids = measure.regionprops(labels)
            areas[r] = [centroid.area for centroid in centroids if (centroid.area > 2000) and (centroid.area < 100000)]
            cells_pos[r] = [centroid.centroid for centroid in centroids if (centroid.area > 2000) and (centroid.area < 100000)]       
            num_cells[r] = len(cells_pos[r])

            cells_to_drop = filter_close_cells(cells_pos[r], areas[r], threshold_distance=threshold_distance)
            cells_pos[r] = [cell for i, cell in enumerate(cells_pos[r]) if i not in cells_to_drop]

            if plot is True:
                _, axes = plt.subplots(nrows=1, ncols=4, figsize=(8, 8), sharex=True, sharey=True)
                ax = axes.ravel()
                ax[0].imshow(binary_image)
                ax[0].set_title("Binary")

                ax[1].imshow(closed_image)
                ax[1].set_title("Closed")

                ax[2].imshow(dilation)
                ax[2].set_title("Dilation")

                ax[3].imshow(roi_image)
                x = [c[1] for c in cells_pos[r]]
                y = [c[0] for c in cells_pos[r]]
                ax[3].scatter(x,y,1,'r')
                ax[3].set_title("Cells")

                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig(output_file + '_steps.png')
                plt.savefig(output_file + '_steps.svg', format='svg', dpi=300)

        elif staining == 'NISSL':
            threshold_distance = 20
            binary_image = rank.gradient(roi_image, disk(2), mask=mask) > 20

            closed_image = remove_small_holes(binary_image, 300, connectivity=2)
            closed_image = remove_small_objects(closed_image, 100, connectivity=2)
            closed_image = closed_image.astype(np.uint8)

            kernel = np.ones((2,2),np.uint8)
            dilation = cv2.dilate(closed_image, kernel, iterations = 1)
            erosion = cv2.erode(dilation, kernel, iterations = 2)
            erosion = remove_small_holes(erosion, 300, connectivity=2)

            # Find cell coords
            labels = ndi.label(erosion)[0]
            centroids = measure.regionprops(labels)
            areas[r] = [centroid.area for centroid in centroids if (centroid.area > 200) and (centroid.area < 100000)]
            cells_pos[r] = [centroid.centroid for centroid in centroids if (centroid.area > 200) and (centroid.area < 100000)]       
            num_cells[r] = len(cells_pos[r])

            cells_to_drop = filter_close_cells(cells_pos[r], areas[r], threshold_distance=threshold_distance)
            cells_pos[r] = [cell for i, cell in enumerate(cells_pos[r]) if i not in cells_to_drop]

            if plot is True:
                _, axes = plt.subplots(nrows=1, ncols=4, figsize=(8, 8), sharex=True, sharey=True)
                ax = axes.ravel()
                ax[0].imshow(binary_image)
                ax[0].set_title("Binary")

                ax[1].imshow(closed_image)
                ax[1].set_title("Closed")

                ax[2].imshow(erosion)
                ax[2].set_title("Erosion")

                ax[3].imshow(roi_image)
                x = [c[1] for c in cells_pos[r]]
                y = [c[0] for c in cells_pos[r]]
                ax[3].scatter(x,y,1,'r')
                ax[3].set_title("Cells")

                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig(output_file + '_steps.png')
                plt.savefig(output_file + '_steps.svg', format='svg', dpi=300)
        
        elif staining == 'GFAP':
            threshold_distance = 40
            binary_image = rank.gradient(roi_image, disk(2), mask=mask) > 10

            closed_image = closing(binary_image, disk(5)) # dilation followed by erosion
            closed_image = remove_small_holes(closed_image, 300, connectivity=2)
            closed_image = remove_small_objects(closed_image, 300, connectivity=2)
            closed_image = closed_image.astype(np.uint8)

            kernel = np.ones((3,3),np.uint8)
            erosion = cv2.erode(closed_image, kernel, iterations = 3)
            dilation = cv2.dilate(erosion, kernel, iterations = 4)

            # Find cell coords
            labels = ndi.label(dilation)[0]
            centroids = measure.regionprops(labels)
            areas[r] = [centroid.area for centroid in centroids if (centroid.area > 500) and (centroid.area < 100000)]
            cells_pos[r] = [centroid.centroid for centroid in centroids if (centroid.area > 500) and (centroid.area < 100000)]       
            num_cells[r] = len(cells_pos[r])

            cells_to_drop = filter_close_cells(cells_pos[r], areas[r], threshold_distance=threshold_distance)
            cells_pos[r] = [cell for i, cell in enumerate(cells_pos[r]) if i not in cells_to_drop]

            if plot is True:
                _, axes = plt.subplots(nrows=1, ncols=4, figsize=(8, 8), sharex=True, sharey=True)
                ax = axes.ravel()
                ax[0].imshow(binary_image)
                ax[0].set_title("Binary")

                ax[1].imshow(closed_image)
                ax[1].set_title("Closed")

                ax[2].imshow(dilation)
                ax[2].set_title("Dilation")

                ax[3].imshow(roi_image)
                x = [c[1] for c in cells_pos[r]]
                y = [c[0] for c in cells_pos[r]]
                ax[3].scatter(x,y,1,'r')
                ax[3].set_title("Cells")

                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig(output_file + '_steps.png')
                plt.savefig(output_file + '_steps.svg', format='svg', dpi=300)

        elif staining == 'DAPI':
            threshold_distance = 20
            binary_image = rank.gradient(roi_image, disk(1), mask=mask) > 12

            kernel = np.ones((2,1),np.uint8)
            binary_image = binary_image.astype(np.uint8)
            erosion = cv2.erode(binary_image, kernel, iterations = 2)

            closed_image = closing(erosion, disk(1))
            closed_image = remove_small_holes(closed_image, 300, connectivity=2)
            closed_image = closed_image.astype(np.uint8)

            labels = ndi.label(closed_image.astype(np.uint8))[0]
            centroids = measure.regionprops(labels)
            areas[r] = [centroid.area for centroid in centroids if (centroid.area > 50) and (centroid.area < 100000)]
            cells_pos[r] = [centroid.centroid for centroid in centroids if (centroid.area > 50) and (centroid.area < 100000)]       
            num_cells[r] = len(cells_pos[r])

            cells_to_drop = filter_close_cells(cells_pos[r], areas[r], threshold_distance=threshold_distance)
            cells_pos[r] = [cell for i, cell in enumerate(cells_pos[r]) if i not in cells_to_drop]

            if plot is True:
                _, axes = plt.subplots(nrows=1, ncols=4, figsize=(8, 8), sharex=True, sharey=True)
                ax = axes.ravel()
                ax[0].imshow(binary_image)
                ax[0].set_title("Binary")

                ax[1].imshow(erosion)
                ax[1].set_title("Erosion")

                ax[2].imshow(closed_image)
                ax[2].set_title("Closed image")

                ax[3].imshow(roi_image)
                x = [c[1] for c in cells_pos[r]]
                y = [c[0] for c in cells_pos[r]]
                ax[3].scatter(x,y,1,'r')
                ax[3].set_title("Cells")
                plt.show()

                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig(output_file + '_steps.png')
                plt.savefig(output_file + '_steps.svg', format='svg', dpi=300)

    # Remove double-counted cells (in two different columns)
    all_cells = []
    all_areas = []
    index_mapping = []
    for r in sorted(cells_pos.keys()):
        for i, cell in enumerate(cells_pos[r]):
            all_cells.append(cell)
            all_areas.append(areas[r][i])
            index_mapping.append((r, i))

    cells_to_drop = filter_close_cells(all_cells, all_areas, threshold_distance=threshold_distance)

    new_cells_pos = {r: [] for r in cells_pos.keys()}
    new_areas = {r: [] for r in areas.keys()}
    new_num_cells = {r: [] for r in areas.keys()}

    for idx, (r, i) in enumerate(index_mapping):
        if idx not in cells_to_drop:
            new_cells_pos[r].append(cells_pos[r][i])
            new_areas[r].append(areas[r][i])
            new_num_cells[r] = len(new_cells_pos[r])

    cells_pos = new_cells_pos
    num_cells = new_num_cells

    # Filter cells according to supplementary staining - centroids need to overlap within a range
    if num_chs_per_cell != 1 and staining_suppl is not None: 
        if staining != staining_suppl:
            index_mapping = [(r, i) for r in sorted(cells_pos.keys()) for i in range(len(cells_pos[r]))]
            centroids1 = np.array([cell for r in sorted(cells_pos.keys()) for (i, cell) in enumerate(cells_pos[r])])
            centroids2 = np.array(cells_pos_suppl)

            keep_indices = []
            for (i, cell) in enumerate(centroids1):
                distances = np.linalg.norm(centroids2 - cell, axis=1)
                if np.any(distances <= 100):
                    keep_indices.append(i)

            new_cells_pos = {r: [] for r in cells_pos.keys()}
            new_num_cells = {r: [] for r in cells_pos.keys()}
            for idx, (r, i) in enumerate(index_mapping):
                if idx in keep_indices:
                    new_cells_pos[r].append(cells_pos[r][i])
                    new_num_cells[r] = len(new_cells_pos[r])

            cells_pos_filt = new_cells_pos
            num_cells_filt = new_num_cells

    return num_cells, cells_pos#, num_cells_filt, cells_pos_filt


def log_array_info(name, array):
    logging.debug(f"{name}: shape={array.shape}, dtype={array.dtype}, size={array.nbytes / (1024 ** 3):.2f} GiB")


def process_radius(shared_image_name, image_shape, radius_image, r, staining, return_dict):
    try:
        logging.debug(f"Process {r} starting.")
        existing_shm = shared_memory.SharedMemory(name=shared_image_name)
        image = np.ndarray(image_shape, dtype=np.uint8, buffer=existing_shm.buf)
        radius_image = np.uint8(radius_image)

        # Select ROI
        roi = np.nonzero(image * radius_image)
        roi_image = np.zeros(image.shape, dtype='uint8')
        roi_image[roi] = image[roi]

        # Create mask for the ROI 
        mask = np.full_like(image, False, dtype=bool)
        mask[roi] = True

        cells_pos = []
        areas = []

        # Count cells 
        if staining == 'IBA':
            binary_image = rank.gradient(roi_image, disk(2), mask=mask) > 20
            closed_image = closing(binary_image, disk(5))
            closed_image = remove_small_holes(closed_image, 300, connectivity=2)
            closed_image = remove_small_objects(closed_image, 500, connectivity=2)
            closed_image = closed_image.astype(np.uint8)
            kernel = np.ones((3, 3), np.uint8)
            dilation = cv2.dilate(closed_image, kernel, iterations=5)
            labels = ndi.label(dilation.astype(np.uint8))[0]

            centroids = measure.regionprops(labels)
            areas.extend([centroid.area for centroid in centroids if (centroid.area > 2000) and (centroid.area < 100000)])
            cells_pos.extend([centroid.centroid for centroid in centroids if (centroid.area > 2000) and (centroid.area < 100000)])       

            cells_to_drop = filter_close_cells(cells_pos, areas, threshold_distance=40)
            cells_pos = [cell for i, cell in enumerate(cells_pos) if i not in cells_to_drop]

        elif staining == 'NISSL':
            binary_image = rank.gradient(roi_image, disk(2), mask=mask) > 20
            closed_image = remove_small_holes(binary_image, 300, connectivity=2)
            closed_image = remove_small_objects(closed_image, 100, connectivity=2)
            closed_image = closed_image.astype(np.uint8)
            kernel = np.ones((2, 2), np.uint8)
            dilation = cv2.dilate(closed_image, kernel, iterations=1)
            erosion = cv2.erode(dilation, kernel, iterations=2)
            erosion = remove_small_holes(erosion.astype(np.uint8), 300, connectivity=2)
            labels = ndi.label(erosion.astype(np.uint8))[0]

            centroids = measure.regionprops(labels)
            areas.extend([centroid.area for centroid in centroids if (centroid.area > 200) and (centroid.area < 100000)])
            cells_pos.extend([centroid.centroid for centroid in centroids if (centroid.area > 200) and (centroid.area < 100000)])       

            cells_to_drop = filter_close_cells(cells_pos, areas, threshold_distance=20)
            cells_pos = [cell for i, cell in enumerate(cells_pos) if i not in cells_to_drop]

        elif staining == 'GFAP':
            binary_image = rank.gradient(roi_image, disk(2), mask=mask) > 20
            closed_image = closing(binary_image, disk(5))
            closed_image = remove_small_holes(closed_image, 300, connectivity=2)
            closed_image = remove_small_objects(closed_image, 300, connectivity=2)
            closed_image = closed_image.astype(np.uint8)
            kernel = np.ones((3, 3), np.uint8)
            erosion = cv2.erode(closed_image, kernel, iterations=3)
            dilation = cv2.dilate(erosion, kernel, iterations=4)
            labels = ndi.label(dilation.astype(np.uint8))[0]

            centroids = measure.regionprops(labels)
            areas.extend([centroid.area for centroid in centroids if (centroid.area > 500) and (centroid.area < 100000)])
            cells_pos.extend([centroid.centroid for centroid in centroids if (centroid.area > 500) and (centroid.area < 100000)])      
           
            cells_to_drop = filter_close_cells(cells_pos, areas, threshold_distance=40)
            cells_pos = [cell for i, cell in enumerate(cells_pos) if i not in cells_to_drop]

        elif staining == 'DAPI':
            binary_image = rank.gradient(roi_image, disk(2), mask=mask) > 20
            closed_image = closing(binary_image, disk(5))
            closed_image = remove_small_holes(closed_image, 300, connectivity=2)
            closed_image = remove_small_objects(closed_image, 300, connectivity=2)
            closed_image = closed_image.astype(np.uint8)
            kernel = np.ones((3, 3), np.uint8)
            erosion = cv2.erode(closed_image, kernel, iterations=3)
            dilation = cv2.dilate(erosion, kernel, iterations=4)
            labels = ndi.label(dilation.astype(np.uint8))[0]

            centroids = measure.regionprops(labels)
            areas.extend([centroid.area for centroid in centroids if (centroid.area > 500) and (centroid.area < 100000)])
            cells_pos.extend([centroid.centroid for centroid in centroids if (centroid.area > 500) and (centroid.area < 100000)])      
           
            cells_to_drop = filter_close_cells(cells_pos, areas, threshold_distance=20)
            cells_pos = [cell for i, cell in enumerate(cells_pos) if i not in cells_to_drop]

        logging.debug(f"Process {r} finished.")  
        return_dict[r] = cells_pos, areas

    except Exception as e:
        logging.error(f"Error in process {r}: {e}")
        return_dict[r] = None

    finally:
        existing_shm.close()


def count_cells_wrapper(image, radius_images, radii, staining, num_chs_per_cell=1, cells_pos_suppl=None, staining_suppl=None):
    '''
    Wrapper to count the number of cells in each bin (radius) using a pool of processes. Double-counted cells in two bins 
    are removed from one of the two bins.

    Note that the parameters here e.g., threshold for markers detection, 
    area of cells etc. have been empirically determined for each staining type. 
    '''
    num_cells = {}
    cells_pos = {}
    areas = {}
    
    # Preprocess image
    image = image.convert('RGB')
    image = img_as_ubyte(np.asarray(image))
    image = biascorrect(image)
    image = img_as_ubyte(gaussian(image, sigma=3))
    
    # Shared memory setup
    shm_image = shared_memory.SharedMemory(create=True, size=image.nbytes)
    manager = Manager()
    return_dict = manager.dict()

    shm_image_np = np.ndarray(image.shape, dtype=np.uint8, buffer=shm_image.buf)
    np.copyto(shm_image_np, image)

    # Count cells in each bin (radius) using a pool of processes
    pool = Pool
    with Pool() as pool:
        pool.starmap(
            process_radius,
            [(shm_image.name, image.shape, radius_images[r], r, staining, return_dict) for r in range(len(radii)*2-2)]
        )
    
    pool.close()
    pool.join()

    for r in return_dict.keys():
        if return_dict[r] is not None:
            cells_pos[r], areas[r] = return_dict[r]
            num_cells[r] = len(cells_pos[r])

    # Remove double-counted cells
    all_cells = []
    all_areas = []
    index_mapping = []
    for r in sorted(cells_pos.keys()):
        for i, cell in enumerate(cells_pos[r]):
            all_cells.append(cell)
            all_areas.append(areas[r][i])
            index_mapping.append((r, i))

    cells_to_drop = filter_close_cells(all_cells, all_areas, threshold_distance=20)

    new_cells_pos = {r: [] for r in cells_pos.keys()}
    new_areas = {r: [] for r in areas.keys()}
    new_num_cells = {r: [] for r in areas.keys()}

    for idx, (r, i) in enumerate(index_mapping):
        if idx not in cells_to_drop:
            new_cells_pos[r].append(cells_pos[r][i])
            new_areas[r].append(areas[r][i])
            new_num_cells[r] = len(new_cells_pos[r])

    cells_pos = new_cells_pos
    num_cells = new_num_cells

    # Filter cells according to supplementary staining
    if num_chs_per_cell != 1 and staining != staining_suppl:
        centroids1 = np.array([(r, cell) for r, cells in cells_pos.values() for cell in cells])
        centroids2 = np.array([(r, cell) for r, cells in cells_pos_suppl.values() for cell in cells])

        keep_indices = []

        for idx, (r, c1) in enumerate(centroids1):
            distances = np.linalg.norm(centroids2 - c1, axis=1)
            if np.any(distances <= 10):
                keep_indices.append(idx)

        new_cells_pos = {r: [] for r in cells_pos.keys()}

        for idx in keep_indices:
            r, c1 = centroids1[idx]
            new_cells_pos[r].append(c1)
            new_num_cells[r] = len(new_cells_pos[r])
        
        cells_pos = new_cells_pos
        num_cells = new_num_cells

    return num_cells, cells_pos
    

def define_roi(image, imarray, line_x, line_y, radii, um_per_pix):
    '''
    Define ROI around the probe and subdivide it into bins (radii)
    '''
    x = np.arange(image.size[0])
    y = np.arange(image.size[1])

    a,b = np.polyfit(line_x, line_y, 1)

    points_in_line = {}
    points_in_radius = {}
    close_points = {}
    for i in tqdm(range(len(line_x))):
        xi = line_x[i]
        y_perp = a*xi + b + (xi-x) / a
        y_perp = np.around(y_perp)
        # points_in_line[i] = np.vstack((x,y_perp)) 
        dist_to_line = np.sqrt((x - line_x[i])**2 + (y_perp-line_y[i])**2) * um_per_pix
        for r in range(1,len(radii)):
            idx1 = r
            idx2 = (len(radii))*2 - r -1
            close_points_idx = np.where(np.logical_and((dist_to_line <= radii[r-1]),(dist_to_line > radii[r])))[0]
            close_points[r] = np.vstack((x[close_points_idx], y_perp[close_points_idx]))

            # Select close points on one side of the line
            dist_to_line_left = ((close_points[r][0] - line_x[0]) * (line_y[-1] - line_y[0]) - (close_points[r][1] - line_y[0]) * (line_x[-1] - line_x[0])) > 0
            dist_to_line_right = ((close_points[r][0] - line_x[0]) * (line_y[-1] - line_y[0]) - (close_points[r][1] - line_y[0]) * (line_x[-1] - line_x[0])) < 0
            left_close_points = close_points[r][:, dist_to_line_left]
            right_close_points = close_points[r][:, dist_to_line_right]
            
            if i == 0:
                points_in_radius[idx1] = left_close_points
                points_in_radius[idx2] = right_close_points
            else: 
                points_in_radius[idx1] = np.hstack((points_in_radius[idx1], left_close_points))
                points_in_radius[idx2] = np.hstack((points_in_radius[idx2], right_close_points))

    # plt.imshow(image)
    # for r in range(1,len(radii)):
    #     plt.plot(points_in_radius[1][0], points_in_radius[1][1],'r')

    radius_images = np.zeros((len(radii)*2-2,imarray.shape[0],imarray.shape[1]),dtype='int8')
    for r in tqdm(range(1,len(radii)*2-1)):
        points_in_radius[r] = points_in_radius[r].astype('int16')
        radius_images[r-1][points_in_radius[r][1],points_in_radius[r][0]] = 1
        # neigh = neigh_ct(radius_images[r])
        # radius_images[r][np.logical_and((neigh > 2),(radius_images[r] == 0))] = 1

    return radius_images


def transform_roi(imarray, line_x, line_y, radii, crop_coords, crop_dims, output_file, cells_pos=None, plot=True):
    '''
    Transform the ROI to align it to the screen. Plot the probe line and optionally the detected cells too. 
    '''
    # Calculate scaling factors
    cropped_image = Image.new('RGB', (crop_dims[0],crop_dims[1]), (255, 255, 255))
    cropped_image_array = np.asarray(cropped_image)

    tl = (crop_coords[0], crop_coords[1])
    tr = (crop_coords[6], crop_coords[7])
    br = (crop_coords[4], crop_coords[5])
    bl = (crop_coords[2], crop_coords[3])
    pts = np.array([tl, bl, br, tr])

    # Find target points in output image w.r.t. the quad
    # Attention: The order must be the same as defined by the roi points!
    tl_dst = (0, 0)
    tr_dst = (crop_dims[0], 0)
    br_dst = (crop_dims[0], crop_dims[1])
    bl_dst = (0, crop_dims[1])
    dst_pts = np.array([tl_dst, bl_dst, br_dst, tr_dst])

    # Get transformation matrix, and warp image
    pts = np.float32(pts.tolist())
    dst_pts = np.float32(dst_pts.tolist())
    M = cv2.getPerspectiveTransform(pts, dst_pts)
    image_size = (cropped_image_array.shape[1], cropped_image_array.shape[0])
    
    imarray = imarray.astype(np.uint8)
    warped = cv2.warpPerspective(imarray, M, dsize=image_size)
    image_new = Image.fromarray(warped)
    
    # Transform the points
    transformed_points = None
    if cells_pos is not None:
        transformed_points = [[] for _ in range(len(radii)*2-2)]
        for r in range(len(radii)*2-2):
            original_points = np.array([[y, x] for x, y in cells_pos[r]], dtype=np.float32)
            transformed_points_array = cv2.perspectiveTransform(original_points.reshape(-1,1,2), M)
            if transformed_points_array is not None:
                transformed_points[r] = [tuple(point[0]) for point in transformed_points_array]

    # Transform the line
    line_homogeneous = np.array([np.column_stack((line_x, line_y))], dtype=np.float32)
    trans_line_array = cv2.perspectiveTransform(line_homogeneous.reshape(-1,1,2), M)
    trans_line = [tuple(point[0]) for point in trans_line_array]

    # Plot detected cells superimposed on cropped figure around the probe.
    if plot is True:
        plt.figure()
        plt.imshow(image_new)
        for r in range(len(radii)*2-2):
            if len(transformed_points[r]) != 0:
                x_coords, y_coords = zip(*transformed_points[r])
                plt.scatter(x_coords, y_coords, 0.2, 'r')
        trans_line_x, trans_line_y = zip(*trans_line)
        plt.plot(trans_line_x, trans_line_y, 'w--')
        plt.xlim(0, crop_dims[0])
        plt.ylim(crop_dims[1], 0)       
        plt.rcParams['svg.fonttype'] = 'none'
        plt.savefig(output_file + '.png')
        plt.savefig(output_file + '.svg', format='svg', dpi=300)
        plt.close()

    if transformed_points is not None:
        return image_new, trans_line, transformed_points
    else:
        return image_new, trans_line

    # plt.imshow(image)
    # for r in range(1, len(radii)):
    #     y_coords, x_coords = zip(*cells_pos[r])
    #     plt.scatter(x_coords, y_coords, 0.1, 'r')
    # plt.show()


def biascorrect(image):
    '''
    Remove biases in the image so that the grayscale distribution across the image is homogeneous.
    '''
    # Note the grayscale values of the corrected image are in a different scale
    # image is numpy array
    plt.imsave('./temp_img.png', image)
    n4_test_img = n4bias(['temp_img.png', "img_corrected.nrrd", 1])
    corrected = sitk.GetArrayFromImage(n4_test_img['corrected_image'])

    min_val = np.min(corrected)
    max_val = np.max(corrected)
    normalized_corrected = ((corrected - min_val) / (max_val - min_val)) * 255
    normalized_corrected = normalized_corrected.astype(np.uint8)
    return normalized_corrected


def n4bias(args):
    if len(args) < 2:
        print(
            "Usage: N4BiasFieldCorrection inputImage "
            + "outputImage [shrinkFactor] [maskImage] [numberOfIterations] "
            + "[numberOfFittingLevels]"
        )
        sys.exit(1)

    inputImage = sitk.ReadImage(args[0], sitk.sitkFloat32)
    image = inputImage

    if len(args) > 4:
        maskImage = sitk.ReadImage(args[4], sitk.sitkUInt8)
    else:
        maskImage = sitk.OtsuThreshold(inputImage, 0, 1, 200)

    shrinkFactor = 1
    if len(args) > 3:
        shrinkFactor = int(args[2])
        if shrinkFactor > 1:
            image = sitk.Shrink(
                inputImage, [shrinkFactor] * inputImage.GetDimension()
            )
            maskImage = sitk.Shrink(
                maskImage, [shrinkFactor] * inputImage.GetDimension()
            )

    corrector = sitk.N4BiasFieldCorrectionImageFilter()

    numberFittingLevels = 4

    if len(args) > 6:
        numberFittingLevels = int(args[6])

    if len(args) > 5:
        corrector.SetMaximumNumberOfIterations(
            [int(args[5])] * numberFittingLevels
        )

    corrected_image = corrector.Execute(image, maskImage)

    log_bias_field = corrector.GetLogBiasFieldAsImage(inputImage)

    corrected_image_full_resolution = inputImage / sitk.Exp(log_bias_field)

    return_images = {"input_image": inputImage,
                     "mask_image": maskImage,
                     "log_bias_field": log_bias_field,
                     "corrected_image": corrected_image}
    return return_images


def get_cell_density(radius_images, radii, num_cells, um_per_pix):
    '''
    Measure cell density and the change in cell_density as % from reference bin
    '''
    cell_density = {}
    perc_cell_density = {}

    for r in range(len(radii)*2-2):
        area = len(np.where(radius_images[r] == 1)[0])
        area_um2 = area * um_per_pix**2
        try:
            cell_density[r] = num_cells[r]/area_um2
        except:
            print('there is an error with r: ' + str(r))

    for r in range(len(radii)*2-2):
        perc_cell_density[r] = float(cell_density[r] / cell_density[0]) * 100
        # The cumulative change in cell density w.r.t. the leftmost radius is computed
        # if r == 0:
        # #     perc_cell_density[r] = float(cell_density[r] / cell_density[r]) * 100
        #     perc_cell_density[r] = float((cell_density[r] - cell_density[r]) / cell_density[r] * 100)
        # else:
        # #     perc_change = float((cell_density[r] - cell_density[r-1]) / cell_density[r-1])
        # #     perc_cell_density[r] = float((perc_cell_density[r-1] + perc_change * perc_cell_density[r-1]))
        #     perc_cell_density[r] = float((cell_density[r] - cell_density[r-1]) / cell_density[r-1] * 100)

    return perc_cell_density


def save_results(output_file, slice_ID, radii, analysis, num_cells=None, cells_pos=None, perc_cell_density=None, fluo=None):
    '''
    For each slice/image analyzed, save the cell density and cell coordinates, or fluorescence intensity for each bin in a csv file
    '''

    results_file = output_file + '_cell_counts.csv'
    if os.path.exists(results_file):
        os.remove(results_file)

    with open(results_file, 'a', newline='') as csvfile:
        if analysis == 'fluorescence':
            fieldnames = ['slice_ID','fluorescence', 'radius']
        elif analysis == 'cell_counting':
            fieldnames = ['slice_ID','num_cells', 'cells_pos', 'cell_density', 'radius']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if csvfile.tell() == 0:
            writer.writeheader()

        idx = len(radii)-2
        for r in range(len(radii)*2-2):
            if analysis == 'fluorescence':
                if r < len(radii):
                    writer.writerow({'slice_ID': slice_ID, 'fluorescence': f"{fluo[r]:.3f}", 'radius': radii[r]})
                else:
                    writer.writerow({'slice_ID': slice_ID, 'fluorescence': f"{fluo[r]:.3f}", 'radius': radii[idx]})
                    idx -= 1
            elif analysis == 'cell_counting':
                if r < len(radii):
                    writer.writerow({'slice_ID': slice_ID, 'num_cells': num_cells[r], 'cells_pos': cells_pos[r], 'cell_density': round(perc_cell_density[r],2), 'radius': radii[r]})
                else:
                    writer.writerow({'slice_ID': slice_ID, 'num_cells': num_cells[r], 'cells_pos': cells_pos[r], 'cell_density': round(perc_cell_density[r],2), 'radius': radii[idx]})
                    idx -= 1


def get_stack_stats(csv_folder, analysis):
    '''
    Get the mean +/- std cell density or fluorescence intensity across all slices for each bin 
    ''' 
    csv_files = glob.glob(os.path.join(csv_folder, '*cell_counts.csv'))

    cell_density_slice = [[] for _ in range(len(csv_files))]
    for i in range(len(csv_files)):
        results_file = csv_files[i]
        with open(results_file, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if analysis == 'fluorescence':
                    cell_density_slice[i].append(row['fluorescence'])
                elif analysis == 'cell_counting':
                    cell_density_slice[i].append(row['cell_density'])
        
    mean_cell_density = np.mean(np.array(cell_density_slice, dtype=float), axis=0)
    std_cell_density = np.std(np.array(cell_density_slice, dtype=float), axis=0)

    return mean_cell_density, std_cell_density
        

def plot_example_slice(file, mean_cell_density, std_cell_density, line_x, line_y, radii, crop_coords, crop_dims, output_file, um_per_pix, dr): 
    '''
    Plot an example slice and superimpose a plot of the cumulative change in cell density or fluorescence intensity 
    '''
    image = Image.open(file)
    image = image.convert('RGB')
    imarray = np.asarray(image)

    image_new, trans_line = transform_roi(imarray, line_x, line_y, radii, crop_coords, crop_dims, output_file, plot=False)

    # Create figure
    fig = go.Figure()

    # Add image
    fig.add_layout_image(
        dict(
            source=image_new,
            xref="x",
            yref="y",
            x=0,
            y=crop_dims[1],  # Top-left corner
            sizex=crop_dims[0],
            sizey=crop_dims[1],
            sizing="stretch",
            layer="below"
        )
    )

    # Add trans_line on the primary y-axis
    trans_line_x, trans_line_y = zip(*trans_line)
    fig.add_trace(go.Scatter(x=trans_line_x, y=trans_line_y, mode='lines', line=dict(color='white', dash='dash'), showlegend=False))

    # Add error bars on the secondary y-axis
    x_values = np.linspace(0 + dr / 2, crop_dims[0] - dr / 2, len(radii) * 2 - 2)
    fig.add_trace(go.Scatter(
        x=x_values,
        y=mean_cell_density,
        mode='markers+lines',
        marker=dict(size=3),  
        error_y=dict(
            type='data',
            array=std_cell_density,
            visible=True,
            color='red',
            thickness=0.5,
            width=0.5
        ),
        line=dict(color='red', width=1),
        yaxis='y2',
        showlegend=False
    ))

    # Update layout to add secondary y-axis
    fig.update_layout(
        xaxis=dict(
            title='Distance from probe (um)',
            range=[0, crop_dims[0]],
            tickmode = 'array',
            tickvals=np.linspace(0, round(crop_dims[0]), 5, dtype=int),
            ticktext=np.linspace(-(round((crop_dims[0] * um_per_pix) / 1000) * 1000) / 2, (round((crop_dims[0] * um_per_pix) / 1000) * 1000) / 2, 5, dtype=int),
            showgrid=False,  # Turn off grid lines
            showline=False,  # Show axis line
            zeroline=False   
        ),
        yaxis=dict(
            title='Pixel Values',
            range=[0, crop_dims[1]],  # Ensure full image height is shown
            #scaleanchor='x',  # Maintain aspect ratio
            scaleratio=1,  # Maintain aspect ratio
            visible=False,
            showgrid=False,  # Turn off grid lines
            showline=False,  # Show axis line
            zeroline=False    
        ),
        yaxis2=dict(
            title='Relative change in cell density (%)',
            overlaying='y',
            side='left',
            range=[-200, 200],
            tickmode='array',
            tickvals=np.linspace(-200, 200, 5),
            ticktext=[f'{int(ytick)}%' for ytick in np.linspace(-200, 200, 5)],
            showgrid=False,  # Turn off grid lines
            showline=False,  # Show axis line
            zeroline=False   
        ),
        width = crop_dims[0]/10 + 100,
        height = crop_dims[1]/10 + 100,
        showlegend=False,
        margin=dict(l=20, r=20, b=20, t=20)  # Remove margin to cover the entire grid
    )

    plot_dims = [um_per_pix*(crop_dims[0]/10 + 100), um_per_pix*(crop_dims[1]/10 + 100)]
    fig.write_image(output_file + '.png')
    fig.write_image(output_file + '.svg', format='svg')
    # fig, ax1 = plt.subplots()
    # # plt.figure()
    # ax1.imshow(image_new, aspect='auto')


    # trans_line_x, trans_line_y = zip(*trans_line)
    # ax2 = ax1.twinx()
    # ax2.plot(trans_line_x, trans_line_y, 'w--')
    # ax2.errorbar(np.linspace(0+ dr/2, crop_dims[0]- dr/2, len(radii)*2-2), mean_cell_density, yerr=std_cell_density, color='r', linewidth=1, elinewidth=1, capsize=2)
    # # ax2.errorbar(np.linspace(0+ dr/2, crop_dims[0]- dr/2, len(radii)*2-2), crop_dims[1]/2 + mean_cell_density, yerr=std_cell_density, color='r', linewidth=1, elinewidth=1, capsize=2)
    # # plt.errorbar(np.linspace(0, crop_dims[0]-crop_dims[0]/(len(radii)*2-2) + dr/2, len(radii)*2-2)+dr/2, crop_dims[1]/2 + mean_cell_density, yerr=std_cell_density, color='r', linewidth=1, elinewidth=1, capsize=2)
    # ax2.set_xlim(0, crop_dims[0])
    # ax2.set_xticks(np.linspace(0, crop_dims[0], 5, dtype=int))
    # ax2.set_xticklabels((np.linspace(0, round((crop_dims[0]*um_per_pix)/1000)*1000, 5, dtype=int)))
    # ax1.yaxis.set_visible(False)


    # ax2.spines['left'].set_position(('outward', 0))
    # ax2.yaxis.set_label_position('left')
    # ax2.yaxis.set_ticks_position('left')
    # ax2.set_ylim(0, 200)
    # ax2.set_yticks(np.linspace(0, 200, 5))

    # ax1.set_ylim(crop_dims[1], 0)
    # ax1.set_aspect(crop_dims[0] / crop_dims[1])
    # # ax2.set_yticklabels([f'{int(ytick)}%' for ytick in np.linspace(0, 200, 5)])
    # # plt.ylim(crop_dims[1], 0)  
    # # plt.yticks(np.linspace(0, crop_dims[1], 5, dtype=int), labels=(np.linspace(0, round((crop_dims[1]*um_per_pix)/1000)*1000, 5, dtype=int)))     
    # plt.rcParams['svg.fonttype'] = 'none'
    # plt.savefig(output_file + '.png')
    # plt.savefig(output_file + '.svg', format='svg')
    # plt.close()


def plot_bin_stats(mean_cell_density, std_cell_density, radii, crop_dims, um_per_pix, dr, output_file):
    '''
    Plot an example slice and superimpose a plot of the cumulative change in cell density or fluorescence intensity 
    '''
    plt.figure()
    plt.errorbar(np.linspace(0, len(radii)*2-2, len(radii)*2-2), mean_cell_density, yerr=std_cell_density, color='r', linewidth=1, elinewidth=1, capsize=2)
    
    plt.xlim(-0.5, len(radii)*2-2+0.5)
    plt.ylim(-200,200)  
    plt.xticks(ticks=np.linspace(0, len(radii)*2-2, 5), labels=np.linspace(-500, 500,5, dtype=int))
    plt.yticks(ticks=np.linspace(-200, 200, 5), labels=np.linspace(-200, 200, 5, dtype=int)) 
    plt.xlabel('Distance from probe (um)')
    plt.ylabel('Relative change in cell density (%)')
    plt.rcParams['svg.fonttype'] = 'none'
    plt.savefig(output_file + '.png')
    plt.savefig(output_file + '.svg', format='svg')
    plt.close()

    # # Create figure
    # fig = go.Figure()

    # # Add error bars on the secondary y-axis
    # x_values = np.linspace(0 + dr / 2, crop_dims[0] - dr / 2, len(radii) * 2 - 2)
    # fig.add_trace(go.Scatter(
    #     x=x_values,
    #     y=mean_cell_density,
    #     mode='markers+lines',
    #     error_y=dict(
    #         type='data',
    #         array=std_cell_density,
    #         visible=True,
    #         color='red'
    #     ),
    #     line=dict(color='red'),
    #     yaxis='y',
    #     showlegend=False
    # ))

    # # Update layout to add secondary y-axis
    # fig.update_layout(
    #     xaxis=dict(
    #         title='Distance from probe (um)',
    #         range=[0, crop_dims[0]],
    #         tickmode = 'array',
    #         tickvals=np.linspace(0, round(crop_dims[0]), 5, dtype=int),
    #         ticktext=np.linspace(-(round((crop_dims[0] * um_per_pix) / 1000) * 1000) / 2, (round((crop_dims[0] * um_per_pix) / 1000) * 1000) / 2, 5, dtype=int),
    #         showgrid=True,  # Turn off grid lines
    #         showline=False,  # Show axis line
    #         zeroline=False   
    #     ),
    #     yaxis=dict(
    #         title='Relative change in cell density (%)',
    #         overlaying='y',
    #         side='left',
    #         range=[-200, 200],
    #         tickmode='array',
    #         tickvals=np.linspace(-200, 200, 5),
    #         ticktext=[f'{int(ytick)}%' for ytick in np.linspace(-200, 200, 5)],
    #         showgrid=True,  # Turn off grid lines
    #         showline=False,  # Show axis line
    #         zeroline=False   
    #     ),
    #     width = crop_dims[0]/10 + 100,
    #     height = crop_dims[1]/10 + 100,
    #     showlegend=False,
    #     margin=dict(l=20, r=20, b=20, t=20)  # Remove margin to cover the entire grid
    # )

    # plot_dims = [um_per_pix*(crop_dims[0]/10 + 100), um_per_pix*(crop_dims[1]/10 + 100)]
    # fig.write_image(os.path.join(output_dir_staining, 'iterative_change_plot.png'))
    # fig.write_image(os.path.join(output_dir_staining, 'iterative_change_plot.svg'), format='svg')
    

def filter_close_cells(cells_pos, areas, threshold_distance):
    '''
    Remove cells that are very close to each other, as this is likely to be an artifact.
    '''
    # Calculate pairwise distances between cell centroids
    cell_centroids = np.array(cells_pos)
    pairwise_distances = squareform(pdist(cell_centroids))

    # Merge cells that are closer than the threshold distance
    merged_cells = []
    visited_pairs = set()
    merge_flag = False
    for i in range(len(cells_pos)):
        for j in range(i + 1, len(cells_pos)):
            # Check if pair (i, j) has already been visited
            if (i, j) in visited_pairs or (j, i) in visited_pairs:
                continue
            
            # Mark the pair (i, j) as visited
            visited_pairs.add((i, j))
            merge_flag = False
            
            if pairwise_distances[i, j] < threshold_distance:
                merge_flag = True 
                merged_cells.append((i, j))
                
        if merge_flag:
            break

    cells_to_keep = []
    cells_to_drop = []
    cell1 = [merged_cells[k][0] for k in range(len(merged_cells))]
    cell2 = [merged_cells[k][1] for k in range(len(merged_cells))]
    for k in range(len(merged_cells)):
        if cell1[k] in cells_to_drop or cell2[k] in cells_to_drop:
            continue
        if areas[cell1[k]] > areas[cell2[k]]:
            cells_to_keep.append(cell1[k])
            cells_to_drop.append(cell2[k]) 
        else:
            cells_to_keep.append(cell2[k])
            cells_to_drop.append(cell1[k])

    cells_to_drop = np.unique(cells_to_drop)
    return cells_to_drop


# %%
