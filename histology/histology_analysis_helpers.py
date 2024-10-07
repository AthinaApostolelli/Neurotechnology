#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Athina Apostolelli - April 2024 
Adapted from Tansel Baran Yasar 

Use python version > 3.10 to get the most updated packages
"""
#%%
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.signal import convolve 
from scipy.stats import stats
from scipy.spatial import Delaunay
from scipy.spatial.distance import pdist, squareform
from scipy.fft import fftn, ifftn, fftshift, ifftshift
from scipy import ndimage as ndi
from skimage import measure, color
from skimage.morphology import disk, closing, remove_small_objects, remove_small_holes, diameter_closing
from skimage.segmentation import watershed
from skimage.color import rgb2gray
from skimage.filters import rank, gaussian, sobel
from skimage.util import img_as_ubyte
from skimage.exposure import histogram
from skimage.feature import peak_local_max
import os, csv, sys, glob, cv2, math
import SimpleITK as sitk
import plotly.graph_objects as go
from numba import jit, cuda
import cupy as cp
import logging
import alphashape # pip install git+https://github.com/bellockk/alphashape.git
from descartes import PolygonPatch
from multiprocessing import Pool, shared_memory, Manager
from shapely.geometry import MultiPoint, Polygon, Point
from shapely.ops import unary_union


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


def count_cells_pyr(image, roi_bin_image, staining, output_file, num_chs_per_cell=1, cells_pos_suppl=None, staining_suppl=None, plot=False):
    '''
    Count the number of cells in the entire ROI. 

    Note that the parameters here e.g., threshold for markers detection, 
    area of cells etc. have been empirically determined for each staining type. 
    '''
    num_cells = {}
    cells_pos = {}
    areas = {}
    # image = image.convert('RGB')
    image_original = np.asarray(image.convert('RGB'))
    image = image.convert('L')
    image = img_as_ubyte(np.asarray(image))
    image = biascorrect(image)
    image = img_as_ubyte(gaussian(image, sigma=1))
    # image = img_as_ubyte(image)

    roi = np.nonzero(image * roi_bin_image)
    roi_image = np.zeros((image.shape[0],image.shape[1]), dtype='uint8')
    roi_image[roi[0],roi[1]] = image[roi]

    roi_image_original = np.zeros_like(image_original, dtype=np.uint8)
    roi_image_original[roi[0],roi[1]] = image_original[roi]

    mask = np.full_like(image, False, dtype=bool)
    mask[roi] = True

    contour_image, labels, markers = preprocess_edges(roi_image, mask, thresholds=[10, 30, 500, 1000, 100000])

    if staining == 'IBA':
        threshold_distance = 20
        binary_image = rank.gradient(roi_image, disk(2), mask=mask) > 10

        kernel = np.ones((3,3),np.uint8)
        erosion = cv2.erode(binary_image.astype(np.uint8), kernel, iterations=2)
        dilation = cv2.dilate(erosion, kernel, iterations=1)

        closed_image = remove_small_holes(dilation, 300, connectivity=2)
        closed_image = remove_small_objects(closed_image, 300, connectivity=2)
        closed_image = closed_image.astype(np.uint8)

        # Find cell coords
        labels = ndi.label(closed_image)[0]
        centroids = measure.regionprops(labels)
        areas = [centroid.area for centroid in centroids if (centroid.area > 300) and (centroid.area < 100000)]
        cells_pos = [centroid.centroid for centroid in centroids if (centroid.area > 300) and (centroid.area < 100000)]       
        if not cells_pos:
            num_cells = 0
        else:
            num_cells = len(cells_pos)
            cells_to_drop = filter_close_cells(cells_pos, areas, threshold_distance=threshold_distance)
            cells_pos = [cell for i, cell in enumerate(cells_pos) if i not in cells_to_drop]

        if plot is True:
            _, axes = plt.subplots(nrows=1, ncols=4, figsize=(8, 8), sharex=True, sharey=True)
            ax = axes.ravel()
            ax[0].imshow(binary_image)
            ax[0].set_title("Binary")

            ax[1].imshow(dilation)
            ax[1].set_title("Dilation")

            ax[2].imshow(closed_image)
            ax[2].set_title("Closed")

            ax[3].imshow(roi_image)
            x = [c[1] for c in cells_pos]
            y = [c[0] for c in cells_pos]
            ax[3].scatter(x,y,1,'r')
            ax[3].set_title("Cells")
            plt.show()

            # plt.rcParams['svg.fonttype'] = 'none'
            # plt.savefig(output_file + '_steps.png')
            # plt.savefig(output_file + '_steps.svg', format='svg', dpi=300)

    elif staining == 'NISSL':
        # threshold_distance = 15

        # # Distance transform 
        # grad_image = rank.gradient(roi_image, disk(2), mask=mask)
        # binary_image = grad_image > 10
        # distance = ndi.distance_transform_edt(binary_image)
        # coords = peak_local_max(distance, footprint=np.ones((2,2)))
        # masking = np.zeros(distance.shape, dtype=bool)
        # masking[tuple(coords.T)] = True
        # markers = ndi.label(masking)[0]
        # labels = watershed(distance, markers, mask=mask)

        # # Distance transform 2 (https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_watershed.html)
        # binary_image = roi_image > 230
        # distance = ndi.distance_transform_edt(binary_image)
        # coords = peak_local_max(distance, footprint=np.ones((2,2)), labels=binary_image)
        # masking = np.zeros(distance.shape, dtype=bool)
        # masking[tuple(coords.T)] = True
        # markers, _ = ndi.label(masking)
        # labels = watershed(-distance, markers, mask=mask, watershed_line=True)

        # # Edge segmentation 
        # grad_image = rank.gradient(roi_image, disk(2), mask=mask)
        # markers = grad_image < 20
        # markers = ndi.label(markers)[0]
        # labels = watershed(grad_image, markers)

        # # Extrema (https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_extrema.html)
        # local_maxima = extrema.local_maxima(roi_image, connectivity=2)
        # label_maxima = measure.label(local_maxima)
        # overlay = color.label2rgb(
        #     label_maxima, roi_image, alpha=0.7, bg_label=0, bg_color=None, colors=[(1, 0, 0)]
        # )

        # h = 5
        # h_maxima = extrema.h_maxima(roi_image, h)
        # label_h_maxima = measure.label(h_maxima)
        # overlay_h = color.label2rgb(
        #     label_h_maxima, roi_image, alpha=0.7, bg_label=0, bg_color=None, colors=[(1, 0, 0)]
        # )

        # fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        # ax[0].imshow(roi_image, cmap='gray')
        # ax[0].set_title('Original image')
        # ax[0].axis('off')

        # ax[1].imshow(overlay)
        # ax[1].set_title('Local Maxima')
        # ax[1].axis('off')

        # ax[2].imshow(overlay_h)
        # ax[2].set_title(f'h maxima for h = {h:.2f}')
        # ax[2].axis('off')
        # plt.show()


        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8, 8), sharex=True, sharey=True)
        ax = axes.ravel()

        ax[0].imshow(contour_image, cmap=plt.cm.gray)
        ax[0].set_title("Binary image")

        ax[1].imshow(markers, cmap=plt.cm.nipy_spectral)
        ax[1].set_title("Markers")

        ax[2].imshow(roi_image_original, cmap=plt.cm.gray)
        ax[2].imshow(labels, cmap=plt.cm.nipy_spectral, alpha=0.5)
        ax[2].set_title("Segmented")

        for a in ax:
            a.axis('off')

        fig.tight_layout()
        plt.show()

        # # Find cell coords
        # labels = ndi.label(closed_image)[0]
        centroids = measure.regionprops(labels)
        cells_pos = [centroid.centroid for centroid in centroids]
        if not cells_pos:
            num_cells = 0
        else:
            num_cells = len(cells_pos)
            # cells_to_drop = filter_close_cells(cells_pos, areas, threshold_distance=threshold_distance)
            # cells_pos = [cell for i, cell in enumerate(cells_pos) if i not in cells_to_drop]

        if plot is True:
            _, ax = plt.subplots()
            ax.imshow(contour_image)
            x = [c[1] for c in cells_pos]
            y = [c[0] for c in cells_pos]
            ax.scatter(x,y,1,'r')
            ax.set_title("Cells")
            plt.show()
    
    elif staining == 'GFAP':
        threshold_distance = 40
        binary_image = rank.gradient(roi_image, disk(2), mask=mask) > 15

        closed_image = closing(binary_image, disk(5)) # dilation followed by erosion
        closed_image = remove_small_holes(closed_image, 300, connectivity=2)
        closed_image = remove_small_objects(closed_image, 100, connectivity=2)
        closed_image = closed_image.astype(np.uint8)

        kernel = np.ones((3,3),np.uint8)
        erosion = cv2.erode(closed_image, kernel, iterations = 3)
        dilation = cv2.dilate(erosion, kernel, iterations = 4)

        # Find cell coords
        labels = ndi.label(dilation)[0]
        centroids = measure.regionprops(labels)
        areas = [centroid.area for centroid in centroids if (centroid.area > 100) and (centroid.area < 100000)]
        cells_pos = [centroid.centroid for centroid in centroids if (centroid.area > 100) and (centroid.area < 100000)]       
        if not cells_pos:
            num_cells = 0
        else:
            num_cells = len(cells_pos)
            cells_to_drop = filter_close_cells(cells_pos, areas, threshold_distance=threshold_distance)
            cells_pos = [cell for i, cell in enumerate(cells_pos) if i not in cells_to_drop]

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
            x = [c[1] for c in cells_pos]
            y = [c[0] for c in cells_pos]
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
        areas = [centroid.area for centroid in centroids if (centroid.area > 50) and (centroid.area < 100000)]
        cells_pos = [centroid.centroid for centroid in centroids if (centroid.area > 50) and (centroid.area < 100000)]       
        if not cells_pos:
            num_cells = 0
        else:
            num_cells = len(cells_pos)
            cells_to_drop = filter_close_cells(cells_pos, areas, threshold_distance=threshold_distance)
            cells_pos = [cell for i, cell in enumerate(cells_pos) if i not in cells_to_drop]

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
            x = [c[1] for c in cells_pos]
            y = [c[0] for c in cells_pos]
            ax[3].scatter(x,y,1,'r')
            ax[3].set_title("Cells")
            plt.show()

            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(output_file + '_steps.png')
            plt.savefig(output_file + '_steps.svg', format='svg', dpi=300)

    # Filter cells according to supplementary staining - centroids need to overlap within a range
    # if num_chs_per_cell != 1 and staining_suppl is not None: 
    #     if staining != staining_suppl:
    #         index_mapping = [(r, i) for r in sorted(cells_pos.keys()) for i in range(len(cells_pos))]
    #         centroids1 = np.array([cell for r in sorted(cells_pos.keys()) for (i, cell) in enumerate(cells_pos)])
    #         centroids2 = np.array(cells_pos_suppl)

    #         keep_indices = []
    #         for (i, cell) in enumerate(centroids1):
    #             distances = np.linalg.norm(centroids2 - cell, axis=1)
    #             if np.any(distances <= 100):
    #                 keep_indices.append(i)

    #         new_cells_pos = {r: [] for r in cells_pos.keys()}
    #         new_num_cells = {r: [] for r in cells_pos.keys()}
    #         for idx, (r, i) in enumerate(index_mapping):
    #             if idx in keep_indices:
    #                 new_cells_pos.append(cells_pos[i])
    #                 new_num_cells = len(new_cells_pos)

    #         cells_pos = new_cells_pos
    #         num_cells = new_num_cells

    return num_cells, cells_pos


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
            if not cells_pos[r]:
                num_cells[r] = 0
            else:
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
            if not cells_pos[r]:
                num_cells[r] = 0
            else:
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
            binary_image = rank.gradient(roi_image, disk(2), mask=mask) > 15

            closed_image = closing(binary_image, disk(5)) # dilation followed by erosion
            closed_image = remove_small_holes(closed_image, 300, connectivity=2)
            closed_image = remove_small_objects(closed_image, 100, connectivity=2)
            closed_image = closed_image.astype(np.uint8)

            kernel = np.ones((3,3),np.uint8)
            erosion = cv2.erode(closed_image, kernel, iterations = 3)
            dilation = cv2.dilate(erosion, kernel, iterations = 4)

            # Find cell coords
            labels = ndi.label(dilation)[0]
            centroids = measure.regionprops(labels)
            areas[r] = [centroid.area for centroid in centroids if (centroid.area > 100) and (centroid.area < 100000)]
            cells_pos[r] = [centroid.centroid for centroid in centroids if (centroid.area > 100) and (centroid.area < 100000)]       
            if not cells_pos[r]:
                num_cells[r] = 0
            else:
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
            if not cells_pos[r]:
                num_cells[r] = 0
            else:
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
            if new_cells_pos[r]:
                new_num_cells[r] = len(new_cells_pos[r])
            else:
                new_num_cells[r] = 0

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

            cells_pos = new_cells_pos
            num_cells = new_num_cells

    return num_cells, cells_pos


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
            binary_image = rank.gradient(roi_image, disk(2), mask=mask) > 10
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

            if cells_pos:
                cells_to_drop = filter_close_cells(cells_pos, areas, threshold_distance=40)
                cells_pos = [cell for i, cell in enumerate(cells_pos) if i not in cells_to_drop]

        elif staining == 'NISSL':
            binary_image = rank.gradient(roi_image, disk(2), mask=mask) > 15
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

            if cells_pos:
                cells_to_drop = filter_close_cells(cells_pos, areas, threshold_distance=20)
                cells_pos = [cell for i, cell in enumerate(cells_pos) if i not in cells_to_drop]

        elif staining == 'GFAP':
            binary_image = rank.gradient(roi_image, disk(2), mask=mask) > 15
            closed_image = closing(binary_image, disk(5)) # dilation followed by erosion
            closed_image = remove_small_holes(closed_image, 300, connectivity=2)
            closed_image = remove_small_objects(closed_image, 100, connectivity=2)
            closed_image = closed_image.astype(np.uint8)
            kernel = np.ones((3,3),np.uint8)
            erosion = cv2.erode(closed_image, kernel, iterations = 3)
            dilation = cv2.dilate(erosion, kernel, iterations = 4)
            labels = ndi.label(dilation)[0]
            centroids = measure.regionprops(labels)
            areas.extend([centroid.area for centroid in centroids if (centroid.area > 100) and (centroid.area < 100000)])
            cells_pos.extend([centroid.centroid for centroid in centroids if (centroid.area > 100) and (centroid.area < 100000)]       )
            
            if cells_pos:
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
           
            if cells_pos:
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
            if not cells_pos[r]:
                num_cells[r] = 0
            else:
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

    for r in return_dict.keys():
        if new_cells_pos[r]:
            new_num_cells[r] = len(new_cells_pos[r])
        else:
            new_num_cells[r] = 0

    cells_pos = new_cells_pos
    num_cells = new_num_cells

    # Filter cells according to supplementary staining
    if num_chs_per_cell != 1 and staining != staining_suppl:
        centroids1 = np.array([(r, cell) for r, cells in cells_pos.values() for cell in cells])
        centroids2 = np.array([(r, cell) for r, cells in cells_pos_suppl.values() for cell in cells])

        keep_indices = []

        for idx, (r, c1) in enumerate(centroids1):
            distances = np.linalg.norm(centroids2 - c1, axis=1)
            if np.any(distances <= 50):
                keep_indices.append(idx)

        new_cells_pos = {r: [] for r in cells_pos.keys()}

        for idx in keep_indices:
            r, c1 = centroids1[idx]
            new_cells_pos[r].append(c1)
            if new_cells_pos[r]:
                new_num_cells[r] = len(new_cells_pos[r])
            else:
                new_num_cells[r] = 0
        
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
                plt.scatter(x_coords, y_coords, 1, 'r')
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


def transform_roi_pyr(imarray, crop_coords, output_file, line_x=None, line_y=None, cells_pos=None, plot=True):
    '''
    Transform the ROI to align it to the screen. Plot the probe line and optionally the detected cells too. 
    '''
    # Calculate scaling factors
    crop_dims = (np.round(np.sqrt(np.square(crop_coords[4]-crop_coords[2]) + np.square(crop_coords[5]-crop_coords[3]))).astype(int), \
                 np.round(np.sqrt(np.square(crop_coords[2]-crop_coords[0]) + np.square(crop_coords[3]-crop_coords[1]))).astype(int))
    
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
        original_points = np.array([[y, x] for x, y in cells_pos], dtype=np.float32)
        transformed_points_array = cv2.perspectiveTransform(original_points.reshape(-1,1,2), M)
        if transformed_points_array is not None:
            transformed_points = [tuple(point[0]) for point in transformed_points_array]

    # Transform the line
    trans_line = None
    if line_x is not None and line_y is not None:
        line_homogeneous = np.array([np.column_stack((line_x, line_y))], dtype=np.float32)
        trans_line_array = cv2.perspectiveTransform(line_homogeneous.reshape(-1,1,2), M)
        trans_line = [tuple(point[0]) for point in trans_line_array]

    # Plot detected cells superimposed on cropped figure around the probe.
    if plot is True:
        plt.figure()
        plt.imshow(image_new)
        if len(transformed_points) != 0:
            x_coords, y_coords = zip(*transformed_points)
            plt.scatter(x_coords, y_coords, 1, 'r')
        if trans_line is not None:
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
        if num_cells[r]:
            cell_density[r] = num_cells[r]/area_um2
        else:
            cell_density[r] = 0

    r_idx = next((key for key, value in cell_density.items() if value > 0), None)
    for r in range(len(radii)*2-2):
        # if cell_density[0] == 0:
        
        perc_cell_density[r] = float((cell_density[r]-cell_density[r_idx]) / cell_density[r_idx]) * 100
        # else:
        #     perc_cell_density[r] = float(cell_density[r] / cell_density[0]) * 100
        # The cumulative change in cell density w.r.t. the leftmost radius is computed
        # if r == 0:
        # #     perc_cell_density[r] = float(cell_density[r] / cell_density[r]) * 100
        #     perc_cell_density[r] = float((cell_density[r] - cell_density[r]) / cell_density[r] * 100)
        # else:
        # #     perc_change = float((cell_density[r] - cell_density[r-1]) / cell_density[r-1])
        # #     perc_cell_density[r] = float((perc_cell_density[r-1] + perc_change * perc_cell_density[r-1]))
        #     perc_cell_density[r] = float((cell_density[r] - cell_density[r-1]) / cell_density[r-1] * 100)

    return perc_cell_density


def get_cell_density_pyr(roi_on_bin_image, roi_off_bin_image, num_cells_on, num_cells_off, um_per_pix):
    '''
    Measure cell density for roi around and away from probe
    '''
    # Measure cell density in an area of 100 um^2 = 10,000 x um_per_pix^2
    scaling = 100*100 / (250*500)
    # area_on = len(np.where(roi_on_bin_image == 1)[0]) * um_per_pix**2 
    # area_off = len(np.where(roi_off_bin_image == 1)[0]) * um_per_pix**2 
    # area_on = 100**2 
    # area_off = 100**2

    cell_density_on = num_cells_on * scaling
    cell_density_off = num_cells_off * scaling
    
    return cell_density_on, cell_density_off


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


def save_results_pyr(output_file, slice_ID, analysis, num_cells_on=None, num_cells_off=None, \
                     cells_pos_on=None, cells_pos_off=None, cell_density_on=None, cell_density_off=None, \
                        fluo_on=None, fluo_off=None):
    '''
    For each hemi of each slice/image analyzed, save the cell density and cell coordinates, 
    or fluorescence intensity of ROIs around and away from probe in a csv file
    '''
    results_file = output_file + '_cell_counts.csv'
    if os.path.exists(results_file):
        os.remove(results_file)

    with open(results_file, 'a', newline='') as csvfile:
        if analysis == 'fluorescence':
            fieldnames = ['slice_ID','fluorescence_on', 'fluorescence_off']
        elif analysis == 'cell_counting':
            fieldnames = ['slice_ID','num_cells_on', 'num_cells_off', 'cells_pos_on', 'cells_pos_off', 'cell_density_on', 'cell_density_off']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if csvfile.tell() == 0:
            writer.writeheader()

        if analysis == 'fluorescence':
            writer.writerow({'slice_ID': slice_ID, 'fluorescence_on': f"{fluo_on:.3f}", 'fluorescence_off': f"{fluo_off:.3f}"})
            
        elif analysis == 'cell_counting':
            writer.writerow({'slice_ID': slice_ID, 'num_cells_on': num_cells_on, 'num_cells_off': num_cells_off, \
                            'cells_pos_on': cells_pos_on, 'cells_pos_off': cells_pos_off, 'cell_density_on': cell_density_on, 'cell_density_off': cell_density_off})


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
        

def get_stack_stats_pyr(csv_folder, analysis):
    '''
    Get the mean +/- std cell density or fluorescence intensity across all slices for each ROI 
    ''' 
    csv_files = glob.glob(os.path.join(csv_folder, '*cell_counts.csv'))

    cell_density_slice_on = [[] for _ in range(len(csv_files))]
    cell_density_slice_off = [[] for _ in range(len(csv_files))]
    for i in range(len(csv_files)):
        results_file = csv_files[i]
        with open(results_file, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if analysis == 'fluorescence':
                    cell_density_slice_on[i].append(row['fluorescence_on'])
                    cell_density_slice_off[i].append(row['fluorescence_off'])
                elif analysis == 'cell_counting':
                    cell_density_slice_on[i].append(row['cell_density_on'])
                    cell_density_slice_off[i].append(row['cell_density_off'])
        
    mean_cell_density_on = np.mean(np.array(cell_density_slice_on, dtype=np.float64), axis=0)
    mean_cell_density_off = np.mean(np.array(cell_density_slice_off, dtype=np.float64), axis=0)
    std_cell_density_on = np.std(np.array(cell_density_slice_on, dtype=np.float64), axis=0)
    std_cell_density_off = np.std(np.array(cell_density_slice_off, dtype=np.float64), axis=0)

    t_test, pvalue = stats.ttest_ind(a=np.array(cell_density_slice_on, dtype=np.float64), b=np.array(cell_density_slice_off, dtype=np.float64), equal_var=True)

    return mean_cell_density_on, mean_cell_density_off, std_cell_density_on, std_cell_density_off, t_test, pvalue


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
    Plot the cumulative change in cell density or fluorescence intensity 
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

    
def plot_on_off_bars(mean_cell_density_on, mean_cell_density_off, std_cell_density_on, std_cell_density_off, pvalue, output_file):
    '''
    Plot the cell density or fluorescence intensity, comparing the ROIs around and away from the probe
    '''
    fig, ax = plt.subplots(figsize=(5,5))
    x_values = [0,0.15]
    ax.bar(x_values, [mean_cell_density_on[0], mean_cell_density_off[0]], yerr=[std_cell_density_on[0], std_cell_density_off[0]], \
           width=0.1, error_kw=dict(elinewidth=2, capsize=5))

    # Plot significance 
    ax_y0, ax_y1 = plt.gca().get_ylim()
    dh = (ax_y1 - ax_y0)/5
    barh = (ax_y1 - ax_y0)/30
    barx = [x_values[0],x_values[0],x_values[1],x_values[1]]
    y = max([mean_cell_density_on[0], mean_cell_density_off[0]]) + dh
    bary = [y, y+barh, y+barh, y]
    mid = ((x_values[0]+x_values[1])/2, y + barh*1.2)

    plt.plot(barx, bary, c='black')
    plt.text(*mid, convert_pvalue_to_asterisks(pvalue), fontsize=14)
    ax.set_xticks(x_values)
    ax.set_xticklabels(["Around\nprobe","Away from\nprobe"], fontsize=14)
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(-4,-4), useMathText=True)
    ax.ticklabel_format(axis='y', style='sci', useMathText=True)

    plt.ylabel('Cell density (100 um^2)', fontsize=14)
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.savefig(output_file + '.png')
    plt.savefig(output_file + '.svg', format='svg')
    plt.close()    


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


def convert_pvalue_to_asterisks(pvalue):
    if pvalue <= 0.0001:
        return "****"
    elif pvalue <= 0.001:
        return "***"
    elif pvalue <= 0.01:
        return "**"
    elif pvalue <= 0.05:
        return "*"
    return "ns"


def preprocess_edges(img, mask, thresholds=[10, 30, 500, 1000, 100000]):
    image = cv2.GaussianBlur(img, (3,3), 0)

    # (1) Get Canny edges and contours
    _, _, contours1, hierarchy = get_canny_contours(image, thresholds=[0.1,1], masked=False)

    # (2) Process and filter large contours
    contours2a = process_large_contours(image, contours1, hierarchy, area_threshold=thresholds, plot_contours=False)

    contours2b, idx_keep = filter_contours_criteria(contours2a, area_threshold=[0, 500])
    
    contours2c = transform_current_contours(image, contours2b)

    # (3) Filter remaining contours 
    contours3 = [contour for c, contour in enumerate(contours1) if c not in idx_keep]

    all_contour_image = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)
    children_image = np.zeros(image.shape[:2], dtype=np.uint8)
    open_children_image = np.zeros(image.shape[:2], dtype=np.uint8)
    closed_children_image = np.zeros(image.shape[:2], dtype=np.uint8)
    parents_image = np.zeros(image.shape[:2], dtype=np.uint8)
    semifinal_image = np.zeros(image.shape[:2], dtype=np.uint8)
    final_image = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)
    keep_image1 = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)
    keep_image2 = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)
    keep_image3 = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)
    final_image_filled = np.zeros(image.shape[:2], dtype=np.uint8)

    output_images = [all_contour_image, children_image, open_children_image, closed_children_image, parents_image, 
        semifinal_image, final_image, keep_image1, keep_image2, keep_image3, final_image_filled]
    
    contours4, images = standard_contour_processing(image, contours3, area_threshold=[10, 30, 500, 1000, 100000], output_images=output_images)

    # (4) Final contours 
    all_contours = []
    for contour in [contours2c, contours4]:
        all_contours.append(contour)

    final_CNT_IMG = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)
    final_CNT_image = np.zeros(image.shape[:2], dtype=np.uint8)
    for contour in all_contours:
        cv2.drawContours(final_CNT_image, contour, -1, 255, -1)
        cv2.drawContours(final_CNT_IMG, contour, -1, [255,0,0], -1)

    contours5, hierarchy5 = cv2.findContours(final_CNT_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    hierarchy5 = hierarchy5[0]

    # (5) Plotting
    _, axes = plt.subplots(nrows=3, ncols=4, sharex=True, sharey=True, figsize=(16,12))
    ax = axes.ravel()

    # List of images
    # images = [
    #     image, dilation, contour_image,
    #     contours2_image, contours4_image, contours5_image,
    #     semifinal_image, final_CNT_image, final_CNT_IMG
    # ]
    titles = ["Gaussian image", "1. All contours after re-detection", "2. Children", "3. Open children",
            "4. Closed children", "5. Parents", "6. All contours",
            "7. Children of all contours", "8. Final contours", 
            "Contours 3 - already closed children", "Contours 5 - newly closed children", 
            "Contours 6 - new from parents"]
    
    # # Corresponding titles
    # titles = [
    #     "Gaussian image", "Edged after dilation", "All first contours",
    #     "Contours 2", "Contours 4", "Contours 5",
    #     "Semifinal", "Contours final", "Final contours"
    # ] 

    bbox = get_bounding_box(mask)
    for i in range(len(images)):  
        if bbox[0] is not None:  
            cropped_image = crop_image(images[i], bbox)
            if "Final contours" in titles[i]:
                ax[i].imshow(crop_image(images[0], bbox), cmap=plt.cm.gray)
                ax[i].imshow(cropped_image)
            else:
                ax[i].imshow(cropped_image, cmap=plt.cm.gray)
        else:  
            ax[i].imshow(images[i], cmap=plt.cm.gray)
        
        ax[i].set_title(titles[i])

    plt.tight_layout()
    plt.savefig('zoomed_in_FINAL_figure.png')
    plt.show()
    plt.close()

    # TODO: watershed 

    return contours5


def standard_contour_processing(image, contours, area_threshold=[10, 30, 500, 1000, 100000], output_images=None):
    
    contours1 = contours
    for c, contour in enumerate(contours1):
        cv2.drawContours(output_images[0], contour, -1, (0,0,255), -1)

    # Mark smaller contours
    temp_contours = []
    keep_idx = []
    for c, contour in enumerate(contours1):
        if len(contour) >= 10:
            if cv2.contourArea(contour) < area_threshold[2]:
                temp_contours.append(contour)
                keep_idx.append(c)

    for c, contour in enumerate(temp_contours):
        cv2.drawContours(output_images[1], contour, -1, 255, -1)

    contours2, _ = cv2.findContours(output_images[1], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # (3) Keep contours based on area, aspect ratio and solidity
    contours3, closed_idx = filter_contours_criteria(contours2, area_threshold=[area_threshold[1], area_threshold[2]])
    for contour in contours3:
        cv2.drawContours(output_images[7], contour, -1, 255, -1)
        
    detected_mask = np.zeros_like(image)
    cv2.fillPoly(detected_mask, contours3, 255)
    diff_masked_region = cv2.bitwise_and(image, image, mask=detected_mask)            
                    
    # (4) Close open contours 
    open_contours = [contour for j, contour in enumerate(contours2) if j not in closed_idx]
    for c in open_contours:
        cv2.drawContours(output_images[2], c, -1, 255, -1)

    # Expand cells that span several small contours
    closed_contours, _ = close_open_contours(image, open_contours)

    temp_image = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(temp_image, closed_contours, 255)

    # Close remaining contours
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
    dilation = cv2.dilate(temp_image, kernel, iterations=2)
    contours4, _ = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # (5) Keep contours based on area and aspect ratio
    contours4, _ = filter_contours_criteria(contours4, area_threshold=[area_threshold[1], area_threshold[2]])

    # Remove overlaps with already closed contours
    filtered_contours = []
    for contour in contours4:
        if not contour_overlap(contour, diff_masked_region, overlap_threshold=0.2):
            filtered_contours.append(contour)
    cv2.fillPoly(output_images[3], filtered_contours, 255)
    contours5, _ = cv2.findContours(output_images[3], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for contour in contours5:
        cv2.drawContours(output_images[7], contour, -1, (255,0,0), -1)

    # (6) Apply contour detection on the remaining large contours
    temp_contours = [contour for j, contour in enumerate(contours1) if j not in keep_idx]
    for contour in temp_contours:
        cv2.drawContours(output_images[4], contour, -1, 255, -1)

    # Several transformations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
    dilation1 = cv2.dilate(output_images[4].copy(), kernel, iterations=1)
    dilation2 = cv2.morphologyEx(dilation1, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours6, _ = cv2.findContours(dilation2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # (7) Keep contours based on area and aspect ratio
    idx_remove = []
    for c, contour in enumerate(contours6):
        if len(contour) >= 5:
            _, (MA,ma), _ = cv2.fitEllipse(contour)
            if MA/ma < 1/2 or MA/ma > 2 or cv2.contourArea(contour) < area_threshold[0] or cv2.contourArea(contour) > area_threshold[2]:
                idx_remove.append(c)

    temp_contours = [contour for j, contour in enumerate(contours6) if j not in idx_remove]
    
    detected_mask = np.zeros_like(image)
    cv2.fillPoly(detected_mask, contours3, 255)
    cv2.fillPoly(detected_mask, contours5, 255)
    diff_masked_region = cv2.bitwise_and(image, image, mask=detected_mask)       

    filtered_contours = []
    for contour in temp_contours:
        if not contour_overlap(contour, diff_masked_region, overlap_threshold=0.2):
            filtered_contours.append(contour)

    temp_image = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(temp_image, filtered_contours, 255)
    contours7, _ = cv2.findContours(temp_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for c, contour in enumerate(contours7):
        cv2.drawContours(output_images[9], contour, -1, (255,0,0), -1)
    
    # (8) Combine all contours into the final mask 
    all_contours = []
    for contour_list in (contours3, contours5, contours7):
        all_contours.extend(contour_list)

    temp_image = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(temp_image, all_contours, 255)

    # TODO: write function that does this          
    contours8, _ = cv2.findContours(temp_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    temp_contours = []
    remove_idx =[]
    for c, contour in enumerate(contours8):
        if cv2.contourArea(contour) > area_threshold[2]:
            detected_mask = np.zeros_like(image)
            cv2.fillPoly(detected_mask, contour, 255)
            diff_masked_region = cv2.bitwise_and(image, image, mask=detected_mask)  
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
            trans1 = cv2.erode(diff_masked_region.copy(), kernel, iterations=1) 
            trans2 = cv2.morphologyEx(trans1, cv2.MORPH_CLOSE, kernel, iterations=1) 
            new_contours, _ = cv2.findContours(trans2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) 
            remove_idx.append(c)
            if len(new_contours) > 0:
                temp_contours.append(new_contours)
    
    all_contours = []
    contours8 = [contour for c, contour in enumerate(contours8) if c not in remove_idx]
    for contour_list in (contours8, temp_contours):
        all_contours.extend(contour_list) 

    cv2.fillPoly(output_images[10], all_contours, 255)
    contours9, hierarchy9 = cv2.findContours(output_images[10], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    hierarchy9 = hierarchy9[0]

    for contour in contours9:
        cv2.drawContours(output_images[5], contour, -1, 255, -1)

    final_contours, _ = filter_contours_criteria(contours9, hierarchy=hierarchy9, area_threshold=[area_threshold[1], area_threshold[2]])

    # Superimpose the original contours with the new detected
    for c, contour in enumerate(contours1):
        cv2.drawContours(output_images[6], contour, -1, (0,0,255), -1)

    cv2.fillPoly(output_images[6], pts=final_contours, color=(0,255,0))
    for contour in final_contours:
        cv2.drawContours(output_images[6], contour, -1, 255, -1)

    processed_contours = []
    for contour in final_contours:
        processed_contours.append(contour)            
    
    return processed_contours, output_images


def preprocess_edges_v0(img, img_org, mask, thresholds, cell_areas):
    image = cv2.GaussianBlur(img, (3,3), 0)

    # Make copies of image for illustration
    contour_image = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)
    contours2_image = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)
    contours3_image = np.zeros(image.shape[:2], dtype=np.uint8)
    contours4_image = np.zeros(image.shape[:2], dtype=np.uint8)
    contours5_image = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)
    filled_image = np.zeros(image.shape[:2], dtype=np.uint8)
    semifinal_image = np.zeros(image.shape[:2], dtype=np.uint8)
    final_CNT_image = np.zeros(image.shape[:2], dtype=np.uint8)
    final_CNT_IMG = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)

    # Get Canny edges and contours
    _, dilation, contours, hierarchy = get_canny_contours(image, thresholds=[0.1,1], masked=False)
    
    for c, contour in enumerate(contours):
        cv2.drawContours(contour_image, [contour], -1, 255, 1)

    # Apply edge detection again on large contours
    updated_contours = process_large_contours(image, contours, hierarchy, area_threshold=[10, 30, 500, 1000, 100000], plot_contours=False)
    contours = updated_contours

    # (2) Keep contours that fulfil criteria: length, area, aspect ratio, non-overlapping
    temp_contours, idx_keep = filter_contours_criteria(contours, area_threshold=[0, 500])
    
    contours2 = transform_current_contours(image, temp_contours)

    # cmap = cm.get_cmap('inferno', len(contours2))
    # norm = Normalize(vmin=0, vmax=len(contours2))
    for contour in contours2: 
        # color = cmap(norm(c)) 
        # color_rgb = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))  
        cv2.drawContours(contours2_image, contour, -1, [255,0,0], -1)

    # (3) Transform remaining contours and re-apply contour detection
    contours3 = [contour for c, contour in enumerate(contours) if c not in idx_keep]
    for contour in contours3:
        cv2.drawContours(contours3_image, contour, -1, 255, -1)

    # Several transformations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
    dilation1 = cv2.dilate(contours3_image.copy(), kernel, iterations=2)
    dilation2 = cv2.morphologyEx(dilation1, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours4, _ = cv2.findContours(dilation2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for contour in contours4:
        cv2.drawContours(contours4_image, contour, -1, 255, -1)

    # Keep contours that fulfill criteria: length, area, aspect ratio, non-overlapping
    contours5, idx_keep = filter_contours_criteria(contours4, area_threshold=[30, 1000])

    # cmap = cm.get_cmap('inferno', len(contours5))
    # norm = Normalize(vmin=0, vmax=len(contours5))
    for c, contour in enumerate(contours5): 
        # color = cmap(norm(c)) 
        # color_rgb = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))  
        cv2.drawContours(contours5_image, contour, -1, [0,255,0], -1)

    # (4) All contours together
    all_contours = []
    for contour_list in (contours2, contours5):
        all_contours.extend(contour_list)
    for contour in all_contours:
        cv2.drawContours(semifinal_image, contour, -1, 255, -1)

    contours6, hierarchy6 = cv2.findContours(semifinal_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    hierarchy6 = hierarchy6[0]
    final_contours, _ = filter_contours_criteria(contours6, hierarchy=hierarchy6, area_threshold=[30,1000])

    final_hierarchy = []
    for c in range(len(final_contours)):
        final_hierarchy.append(np.array([c-1, c+1, -1, 0], dtype=np.int32)) # no children, parent is first contour in image

    for c, contour in enumerate(final_contours):
        cv2.drawContours(final_CNT_image, contour, -1, 255, -1)
        cv2.drawContours(final_CNT_IMG, contour, -1, (255,0,0), -1)


    # (5) Plotting
    _, axes = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True, figsize=(12,12))
    ax = axes.ravel()

    # List of images
    images = [
        image, dilation, contour_image,
        contours2_image, contours4_image, contours5_image,
        semifinal_image, final_CNT_image, final_CNT_IMG
    ]

    # Corresponding titles
    titles = [
        "Gaussian image", "Edged after dilation", "All first contours",
        "Contours 2", "Contours 4", "Contours 5",
        "Semifinal", "Contours final", "Final contours"
    ]

    bbox = get_bounding_box(mask)
    for i in range(len(images)):  
        if bbox[0] is not None:  
            cropped_image = crop_image(images[i], bbox)
            if "Final contours" in titles[i]:
                ax[i].imshow(crop_image(images[0], bbox), cmap=plt.cm.gray)
                ax[i].imshow(cropped_image)
            else:
                ax[i].imshow(cropped_image, cmap=plt.cm.gray)
        else:  
            ax[i].imshow(images[i], cmap=plt.cm.gray)
        
        ax[i].set_title(titles[i])

    plt.tight_layout()
    plt.savefig('zoomed_in_FINAL_figure.png')
    plt.show()
    plt.close()


    # (6) Final pre-processing - define markers
    ## SKIMAGE
    for contour in final_contours:
        cv2.drawContours(filled_image, [contour], -1, 255, thickness=-1)
    # filled_image = cv2.fillPoly(filled_image, pts=final_contours, color=255)
    # filled_image = cv2.drawContours(filled_image, final_contours, -1, color=255, thickness=cv2.FILLED)
    _, markers = cv2.connectedComponents(filled_image)

    labels = watershed(filled_image, markers, mask=mask, watershed_line=True)
    # distance = ndi.distance_transform_edt(filled_image)
    # coords = peak_local_max(distance, footprint=np.ones((2,2)), labels=filled_image)
    # mask = np.zeros(distance.shape, dtype=bool)
    # mask[tuple(coords.T)] = True
    # markers, _ = ndi.label(mask)
    # labels = watershed(-distance, markers, mask=mask, watershed_line=True)


    ## OPENCV 
    # Sure background and foreground
    # opening = cv2.morphologyEx(filled_image, cv2.MORPH_OPEN, kernel, iterations=2)
    # sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # distance = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    # _, sure_fg = cv2.threshold(distance, 0.7*distance.max(), 255, 0)
    
    # # Finding unknown region
    # sure_fg = np.uint8(sure_fg)
    # unknown = cv2.subtract(sure_bg, sure_fg)

    # _, markers = cv2.connectedComponents(sure_fg)
    # markers = markers+1 # sure bg is 1, not 0 
    
    # # Now, mark the region of unknown with zero
    # markers[unknown==255] = 0
    # labels = watershed(filled_image, markers, mask=mask, watershed_line=True)

    return filled_image, labels, markers


def get_canny_contours(image, thresholds=[0.5,1.5], masked=False, retrieval=cv2.RETR_TREE):
    # Detect edges
    if masked:
        adaptive_thres, _ = cv2.threshold(image[image!=0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edges = cv2.Canny(image, thresholds[0]*adaptive_thres, thresholds[1]*adaptive_thres, L2gradient=True, apertureSize=5)
    else:
        adaptive_thres, _ = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edges = cv2.Canny(image, thresholds[0]*adaptive_thres, thresholds[1]*adaptive_thres, L2gradient=True, apertureSize=3)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
    dilation = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Detect contours 
    # Hierarchy is organised as: [Next, Previous, First_Child, Parent]
    contours, hierarchy = cv2.findContours(dilation, retrieval, cv2.CHAIN_APPROX_NONE)
    hierarchy = hierarchy[0]
    print(len(contours), "objects were found in this image.")

    return edges, dilation, contours, hierarchy
       

def process_large_contours(image, contours, hierarchy, area_threshold=[10, 30, 500, 1000, 100000], plot_contours=False):

    updated_contours = []
    # updated_hierarchy = []
    
    num_new_contours = 0
    counter = 0 # where the big contour was split

    for i, (original_contour, hier) in enumerate(zip(contours, hierarchy)):

        if area_threshold[3] < cv2.contourArea(original_contour) < area_threshold[4]:
            counter += 1

            all_contour_image = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)
            children_image = np.zeros(image.shape[:2], dtype=np.uint8)
            open_children_image = np.zeros(image.shape[:2], dtype=np.uint8)
            closed_children_image = np.zeros(image.shape[:2], dtype=np.uint8)
            parents_image = np.zeros(image.shape[:2], dtype=np.uint8)
            semifinal_image = np.zeros(image.shape[:2], dtype=np.uint8)
            final_image = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)
            keep_image1 = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)
            keep_image2 = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)
            keep_image3 = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)
            final_image_filled = np.zeros(image.shape[:2], dtype=np.uint8)

            output_images = [all_contour_image, children_image, open_children_image, closed_children_image, parents_image, 
              semifinal_image, final_image, keep_image1, keep_image2, keep_image3, final_image_filled]

            # (1) Another round of edge and contour detection with different thresholds
            contour_mask = np.zeros_like(image)
            cv2.drawContours(contour_mask, [original_contour], -1, 255, -1)
            masked_region = cv2.bitwise_and(image, image, mask=contour_mask) 

            _, _, contours1, _ = get_canny_contours(masked_region, thresholds=[0.5,1.5], masked=True, retrieval=cv2.RETR_TREE)
            
            processed_contours, images = standard_contour_processing(image, contours1, area_threshold=area_threshold, output_images=output_images)
            for contour in processed_contours:
                updated_contours.append(contour)

            # for c, contour in enumerate(contours1):
            #     cv2.drawContours(all_contour_image, contour, -1, (0,0,255), -1)

            # # (2) Mark smaller contours
            # temp_contours = []
            # keep_idx = []
            # for c, contour in enumerate(contours1):
            #     if len(contour) >= 10:
            #         if cv2.contourArea(contour) < area_threshold[2]:
            #             temp_contours.append(contour)
            #             keep_idx.append(c)

            # for c, contour in enumerate(temp_contours):
            #     cv2.drawContours(children_image, contour, -1, 255, -1)

            # contours2, _ = cv2.findContours(children_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            # # (3) Keep contours based on area, aspect ratio and solidity
            # contours3, closed_idx = filter_contours_criteria(contours2, area_threshold=[area_threshold[1], area_threshold[2]])
            # for contour in contours3:
            #     cv2.drawContours(keep_image1, contour, -1, 255, -1)
                
            # detected_mask = np.zeros_like(image)
            # cv2.fillPoly(detected_mask, contours3, 255)
            # diff_masked_region = cv2.bitwise_and(image, image, mask=detected_mask)            
                            
            # # (4) Close open contours 
            # open_contours = [contour for j, contour in enumerate(contours2) if j not in closed_idx]
            # for c in open_contours:
            #     cv2.drawContours(open_children_image, c, -1, 255, -1)

            # # Expand cells that span several small contours
            # closed_contours, _ = close_open_contours(image, open_contours)

            # temp_image = np.zeros(image.shape[:2], dtype=np.uint8)
            # cv2.fillPoly(temp_image, closed_contours, 255)

            # # Close remaining contours
            # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
            # dilation = cv2.dilate(temp_image, kernel, iterations=2)
            # contours4, _ = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            # # (5) Keep contours based on area and aspect ratio
            # contours4, _ = filter_contours_criteria(contours4, area_threshold=[area_threshold[1], area_threshold[2]])

            # # Remove overlaps with already closed contours
            # filtered_contours = []
            # for contour in contours4:
            #     if not contour_overlap(contour, diff_masked_region, overlap_threshold=0.2):
            #         filtered_contours.append(contour)
            # cv2.fillPoly(closed_children_image, filtered_contours, 255)
            # contours5, _ = cv2.findContours(closed_children_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            # for contour in contours5:
            #     cv2.drawContours(keep_image2, contour, -1, (255,0,0), -1)

            # # (6) Apply contour detection on the remaining large contours
            # temp_contours = [contour for j, contour in enumerate(contours1) if j not in keep_idx]
            # for contour in temp_contours:
            #     cv2.drawContours(parents_image, contour, -1, 255, -1)

            # # Several transformations
            # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
            # dilation1 = cv2.dilate(parents_image.copy(), kernel, iterations=1)
            # dilation2 = cv2.morphologyEx(dilation1, cv2.MORPH_CLOSE, kernel, iterations=1)
            # contours6, _ = cv2.findContours(dilation2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            # # (7) Keep contours based on area and aspect ratio
            # idx_remove = []
            # for c, contour in enumerate(contours6):
            #     if len(contour) >= 5:
            #         _, (MA,ma), _ = cv2.fitEllipse(contour)
            #         if MA/ma < 1/2 or MA/ma > 2 or cv2.contourArea(contour) < area_threshold[0] or cv2.contourArea(contour) > area_threshold[2]:
            #             idx_remove.append(c)

            # temp_contours = [contour for j, contour in enumerate(contours6) if j not in idx_remove]
            
            # detected_mask = np.zeros_like(image)
            # cv2.fillPoly(detected_mask, contours3, 255)
            # cv2.fillPoly(detected_mask, contours5, 255)
            # diff_masked_region = cv2.bitwise_and(image, image, mask=detected_mask)       

            # filtered_contours = []
            # for contour in temp_contours:
            #     if not contour_overlap(contour, diff_masked_region, overlap_threshold=0.2):
            #         filtered_contours.append(contour)

            # temp_image = np.zeros(image.shape[:2], dtype=np.uint8)
            # cv2.fillPoly(temp_image, filtered_contours, 255)
            # contours7, _ = cv2.findContours(temp_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            # for c, contour in enumerate(contours7):
            #     cv2.drawContours(keep_image3, contour, -1, (255,0,0), -1)
            
            # # (8) Combine all contours into the final mask 
            # all_contours = []
            # for contour_list in (contours3, contours5, contours7):
            #     all_contours.extend(contour_list)

            # temp_image = np.zeros(image.shape[:2], dtype=np.uint8)
            # cv2.fillPoly(temp_image, all_contours, 255)

            # # TODO: write function that does this          
            # contours8, _ = cv2.findContours(temp_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            # temp_contours = []
            # remove_idx =[]
            # for c, contour in enumerate(contours8):
            #     if cv2.contourArea(contour) > area_threshold[2]:
            #         detected_mask = np.zeros_like(image)
            #         cv2.fillPoly(detected_mask, contour, 255)
            #         diff_masked_region = cv2.bitwise_and(image, image, mask=detected_mask)  
            #         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
            #         trans1 = cv2.erode(diff_masked_region.copy(), kernel, iterations=1) 
            #         trans2 = cv2.morphologyEx(trans1, cv2.MORPH_CLOSE, kernel, iterations=1) 
            #         new_contours, _ = cv2.findContours(trans2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) 
            #         remove_idx.append(c)
            #         if len(new_contours) > 0:
            #             temp_contours.append(new_contours)
            
            # all_contours = []
            # contours8 = [contour for c, contour in enumerate(contours8) if c not in remove_idx]
            # for contour_list in (contours8, temp_contours):
            #     all_contours.extend(contour_list) 

            # cv2.fillPoly(final_image_filled, all_contours, 255)
            # contours9, hierarchy9 = cv2.findContours(final_image_filled, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            # hierarchy9 = hierarchy9[0]

            # for contour in contours9:
            #     cv2.drawContours(semifinal_image, contour, -1, 255, -1)

            # final_contours, _ = filter_contours_criteria(contours9, hierarchy=hierarchy9, area_threshold=[area_threshold[1], area_threshold[2]])

            # # final_hierarchy = []
            # # for c in range(len(final_contours)):
            # #     final_hierarchy.append(np.array([c-1, c+1, -1, 0], dtype=np.int32)) # no children, parent is first contour in image
                    
            # # new_idx_hierarchy = [np.where(array != -1, array + i + num_new_contours, array) for array in final_hierarchy]

            # # Superimpose the original contours with the new detected
            # for c, contour in enumerate(contours1):
            #     cv2.drawContours(final_image, contour, -1, (0,0,255), -1)

            # cv2.fillPoly(final_image, pts=final_contours, color=(0,255,0))
            # for contour in final_contours:
            #     cv2.drawContours(final_image, contour, -1, 255, -1)

            # for contour in final_contours:
            #     updated_contours.append(contour)
            # num_new_contours += len(final_contours)   

            ### Plotting ###
            if plot_contours:
                _, axes = plt.subplots(nrows=3, ncols=4, sharex=True, sharey=True, figsize=(16,12))
                ax = axes.ravel()

                # List of images
                # images = [
                #     image, all_contour_image, children_image, open_children_image,
                #     closed_children_image, parents_image, final_image_filled,
                #     semifinal_image, final_image, keep_image1, keep_image2, keep_image3
                # ]

                # Corresponding titles
                titles = [
                    "", "1. All contours after re-detection", "2. Children", "3. Open children",
                    "4. Closed children", "5. Parents", "6. All contours",
                    "7. Children of all contours", "8. Final contours", 
                    "Contours 3 - already closed children", "Contours 5 - newly closed children", 
                    "Contours 6 - new from parents"
                ]

                mask = np.full_like(image, False, dtype=bool)
                contour_x = [p[0][0] for p in original_contour]
                contour_y = [p[0][1] for p in original_contour]
                mask[contour_y, contour_x] = True

                bbox = get_bounding_box(mask)
                for i in range(len(images)):  
                    if bbox[0] is not None:  
                        cropped_image = crop_image(images[i], bbox)
                        if "Final contours" in titles[i]:
                            ax[i].imshow(crop_image(images[0], bbox), cmap=plt.cm.gray)
                            ax[i].imshow(cropped_image)
                        else:
                            ax[i].imshow(cropped_image, cmap=plt.cm.gray)
                    else:  
                        ax[i].imshow(images[i], cmap=plt.cm.gray)
                    
                    ax[i].set_title(titles[i])

                plt.tight_layout()
                plt.savefig('zoomed_in_figure_' + str(counter) + '.png')
                plt.show()            

        else:
            updated_contours.append(original_contour)

            # Update hierarchy with correct indices
            # new_idx_hierarchy = np.where(hier != -1, hier + num_new_contours - 1, hier)
            # updated_hierarchy.append(new_idx_hierarchy)

    return updated_contours


def get_bounding_box(image):
    y_indices, x_indices = np.where(image > 0)
    if len(x_indices) == 0 or len(y_indices) == 0:  # handle cases where the mask is empty
        return None, None, None, None
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()
    return x_min, x_max, y_min, y_max


def crop_image(image, bbox):
    x_min, x_max, y_min, y_max = bbox
    return image[y_min:y_max+1, x_min:x_max+1]


def remove_overlapping_contours(contours):
    centroids = []
    distances = []
    for contour in contours:
        if len(contour) > 1:
            M = cv2.moments(contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroids.append((cX, cY))

        idx_remove = set()
        for k, (cx1, cy1) in enumerate(centroids):
            for j, (cx2, cy2) in enumerate(centroids):
                if k != j and j not in idx_remove:
                    dist = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
                    distances.append(dist)
                    if dist < 5:
                        # Remove larger contour
                        if cv2.contourArea(contours[k]) > cv2.contourArea(contours[j]):
                            idx_remove.add(k)
                        else:
                            idx_remove.add(j)

    return idx_remove


def transform_current_contours(image, contours):

    temp_image = np.zeros(image.shape[:2], dtype=np.uint8)
    for contour in contours:
        cv2.drawContours(temp_image, contour, -1, 255, -1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
    dilation = cv2.dilate(temp_image, kernel, iterations=2)

    new_contours, _ = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    return new_contours


def filter_contours_criteria(contours, hierarchy=None, area_threshold=[40,500]):
    new_contours = []
    closed_idx = []

    if hierarchy is not None:    
        for c, (contour,hier) in enumerate(zip(contours, hierarchy)):
            area = cv2.contourArea(contour)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if len(contour) >= 5:
                _, (MA,ma), _ = cv2.fitEllipse(contour)
                if hull_area != 0:
                    solidity = float(area)/hull_area
                    if area_threshold[0] < area < area_threshold[1] and (1/4 < MA/ma < 4) and solidity > 0.5 and hier[2] == -1:
                        new_contours.append(contour)
                        closed_idx.append(c)

    else:
        areas = []
        sol = []
        asp =[]
        for c, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if len(contour) >= 5:
                _, (MA,ma), _ = cv2.fitEllipse(contour)
                if hull_area != 0:
                    solidity = float(area)/hull_area
                    areas.append(area)
                    sol.append(solidity)
                    asp.append(MA/ma)
                    if area_threshold[0] < area < area_threshold[1] and (1/4 < MA/ma < 4) and solidity > 0.5:
                        new_contours.append(contour)
                        closed_idx.append(c)

    return new_contours, closed_idx


def close_open_contours(image, contours):
    distances = []
    contour_sizes = []
    contours_to_dilate = []
    closed_contours = []

    for i in range(len(contours)):
        contour_sizes.append(len(contours[i]))

        # Get the centroid of contour i
        M1 = cv2.moments(contours[i])
        if M1['m00'] != 0:  # Avoid division by zero
            cX1 = int(M1['m10'] / M1['m00'])
            cY1 = int(M1['m01'] / M1['m00'])
            centroid1 = (cX1, cY1)
        else:
            continue
        
        # Compare with all other contours
        for j in range(i + 1, len(contours)):
            # Get the centroid of contour j
            M2 = cv2.moments(contours[j])
            if M2['m00'] != 0:
                cX2 = int(M2['m10'] / M2['m00'])
                cY2 = int(M2['m01'] / M2['m00'])
                centroid2 = (cX2, cY2)
            else:
                continue
            
            # Calculate the Euclidean distance between centroid1 and centroid2
            dist = math.sqrt((centroid1[0] - centroid2[0])**2 + (centroid1[1] - centroid2[1])**2)
            distances.append(((i, j), dist))

    for dist, size in zip(distances, contour_sizes):
        contour1 = dist[0][0]
        contour2 = dist[0][1]
        if dist[1] < 30 and contour_sizes[contour1] < 20 and contour_sizes[contour2] < 20:
            contours_to_dilate.append(contour1)
            contours_to_dilate.append(contour2)

    closed_contours = [contour for c, contour in enumerate(contours) if c not in contours_to_dilate]

    temp_contours = [contour for c, contour in enumerate(contours) if c in contours_to_dilate]
    temp_image = np.zeros(image.shape[:2], dtype=np.uint8)
    for contour in temp_contours:
        cv2.drawContours(temp_image, contour, -1, 255, -1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
    dilation = cv2.dilate(temp_image, kernel, iterations=3)
    new_contours, _ = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for contour in new_contours:
        closed_contours.append(contour)

    temp_image = np.zeros(image.shape[:2], dtype=np.uint8)
    for contour in closed_contours:
        cv2.drawContours(temp_image, contour, -1, 255, -1)

    closed_contours, closed_hierarchy = cv2.findContours(temp_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    closed_hierarchy = closed_hierarchy[0]

    return closed_contours, closed_hierarchy


def contour_overlap(contour, mask, overlap_threshold=0.4):
    """
    Check if all points of the contour are outside the mask
    """
    total_points = len(contour)
    inside_points = 0

    for point in contour:
        x, y = point[0] 
        if mask[y, x] > 0:  
            inside_points += 1  

    overlap_percentage = inside_points / total_points

    return overlap_percentage >= overlap_threshold

###### OLD VERSION 
# def process_large_contours(image, contours, hierarchy, area_threshold=[1000,10000]):

#     updated_contours = []
#     updated_hierarchy = []

#     filtered_image = np.zeros(image.shape[:2], dtype=np.uint8)
#     closed_parents_image = np.zeros(image.shape[:2], dtype=np.uint8)

#     counter = 0 # where the big contour was split
#     for i, (contour, hier) in enumerate(zip(contours, hierarchy)):

#         if area_threshold[3] < cv2.contourArea(contour) < area_threshold[4]:
#             counter += 1 

#             # (2) Another round of edge and contour detection with different thresholds
#             contour_mask = np.zeros_like(image)
#             cv2.drawContours(contour_mask, [contour], -1, 255, -1)
#             masked_region = cv2.bitwise_and(image, image, mask=contour_mask) 

#             _, _, contours1, hierarchy1 = get_canny_contours(masked_region, thresholds=[0.5,2], masked=True)
#             for n in contours1:
#                 cv2.drawContours(filtered_image, n, -1, 255, -1)
            
#             # (3) Close open contours
#             kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
#             dilation = cv2.morphologyEx(filtered_image, cv2.MORPH_CLOSE, kernel, iterations=3)
#             contours6, hierarchy7 = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                        
#             # (4) Find contour with largest area 
#             max_area = 0
#             max_contour = None
#             idx_remove = []
#             for c, contour in enumerate(contours6):
#                 cv2.drawContours(filtered_image, contour, -1, 255, -1)
#                 area = cv2.contourArea(contour) # area of child
#                 if area > area_threshold[3]:
#                     idx_remove.append(c)                   
#                 if area > max_area:
#                     max_area = area
#                     max_contour = contour 

#             contours6 = [contour for i, contour in enumerate(contours6) if i not in idx_remove]
#             hierarchy7 = np.delete(hierarchy7[0], idx_remove, axis=0)
#             new_idx_hierarchy = [np.where(array != -1, array + i, array) for array in hierarchy7]
#             for j, (contour, hier) in enumerate(zip(contours6, new_idx_hierarchy)):
#                 updated_contours.append(contour)
#                 updated_hierarchy.append(hier)
#                 cv2.drawContours(closed_parents_image, contour, -1, 255, -1)

#             _, axes = plt.subplots(nrows=1,ncols=4, sharex=True, sharey=True)
#             ax = axes.ravel()
#             ax[0].imshow(image)
#             ax[1].imshow(filtered_image, cmap=plt.cm.gray)
#             ax[1].set_title("High thres & closed contours")
#             ax[2].imshow(closed_parents_image, cmap=plt.cm.gray)
#             ax[2].set_title("Final contours part 1")
#             plt.show()

#             # Another round of contour detection within the max contour
#             max_area = 0
#             max_contour = None
#             idx_remove = []
#             for c, contour in enumerate(contours6):
#                 cv2.drawContours(filtered_image, contour, -1, 255, -1)
#                 area = cv2.contourArea(contour)    
#                 if 500 < area:
#                     max_area = area
#                     max_contour = contour # assume there is 1

#             if max_contour is not None:
#                 contour_mask = np.zeros_like(image)
#                 cv2.drawContours(contour_mask, [max_contour], -1, 255, -1)
#                 masked_region = cv2.bitwise_and(image, image, mask=contour_mask) 

#                 _, _, contours8, hierarchy8 = get_canny_contours(masked_region, thresholds=[0.5,2], masked=True)
            
#                 for c in contours8:
#                     cv2.drawContours(closed_parents_image, c, -1, 255, -1)
#                 ax[3].imshow(closed_parents_image, cmap=plt.cm.gray)
#                 ax[3].set_title("Final contours part 1 & 2")
#                 plt.show()

#             # (5) Create bounding polygon around max contour
#             points = [tuple(point[0]) for point in max_contour]
#             alpha = 0.95 * alphashape.optimizealpha(points) # find optimal alpha
#             hull = alphashape.alphashape(points, alpha) 
#             hull_pts = hull.exterior.coords.xy

#             _, axes = plt.subplots(nrows=1, ncols=3, figsize=(8, 8), sharex=True, sharey=True)
#             ax = axes.ravel()
#             if hull.geom_type == 'MultiPolygon':
#                 geoms = [g for g in hull.geoms]
#                 cmap = cm.get_cmap('inferno', len((geoms)))
#                 norm = Normalize(vmin=0,vmax=len(geoms))
#                 for i in range(len(geoms)):
#                     color = cmap(norm(i))  # Get the color for the current contour
#                     color_rgb = ((color[0]), (color[1]), (color[2])) 
#                     plotpts = geoms[i].exterior.coords.xy
#                     ax[0].imshow(image)
#                     ax[0].scatter(*zip(*points),s=1,color='blue')
#                     ax[0].plot(plotpts[0],plotpts[1],color='red')
#                     ax[0].add_patch(PolygonPatch(geoms[i], fill=True, color=color_rgb,alpha=0.8))
#             else:
#                 ax[0].imshow(image)
#                 ax[0].scatter(*zip(*points), s=1, color='blue', label='Original Points')
#                 ax[0].plot(hull_pts[0], hull_pts[1], color='red', label='Alpha Shape')
#                 ax[0].add_patch(PolygonPatch(hull, fill=False, color='green', label='Hull'))
#                 ax[0].legend()

#             # (6) Keep enclosed points only
#             filtered_points = [p for p in points if Point(p).within(hull.buffer(-2))]  
#             ax[1].imshow(image)
#             ax[1].scatter(*zip(*filtered_points), s=1, color='blue', label='Original Points')
#             ax[1].plot(hull_pts[0], hull_pts[1], color='red', label='Alpha Shape')
#             ax[1].legend()

#             # (7) Find contours using enclosed points
#             if filtered_points:
#                 x_filt = [point[0] for point in filtered_points]
#                 y_filt = [point[1] for point in filtered_points]
#                 closed_parents_image[y_filt, x_filt] = 255

#             kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
#             dilation = cv2.morphologyEx(closed_parents_image, cv2.MORPH_CLOSE, kernel, iterations=1)
#             contours8, hierarchy8 = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

#             for c in contours8:
#                 cv2.drawContours(closed_parents_image, c, -1, 255, -1)
#             ax[2].imshow(closed_parents_image, cmap=plt.cm.gray)
#             ax[2].set_title("Final contours part 1 & 2")
#             plt.show()

#             updated_contours.extend(contours8)

#             # (8) Update hierarchy with correct indices
#             # i is the contour we are splitting
#             new_idx_hierarchy = [np.where(array != -1, array + i + len(contours6), array) for array in hierarchy8]
#             for j in new_idx_hierarchy:
#                 updated_hierarchy.extend(j)

#         else:
#             updated_contours.append(contour)

#             # Update hierarchy with correct indices
#             new_idx_hierarchy = np.where(hier != -1, hier + counter, hier)
#             updated_hierarchy.append(new_idx_hierarchy)

#     return updated_contours, updated_hierarchy


### OLD WORKING VERSION 
# def preprocess_edges(img, thresholds, cell_areas):
#     image = cv2.GaussianBlur(img, (3,3), 0)
#     # image = cv2.bilateralFilter(img, d=5, sigmaColor=10, sigmaSpace=10)

#     # Get Canny edges and contours
#     edges, dilation, contours, hierarchy = get_canny_contours(image, thresholds=[0.1,1], masked=False)

#     # Apply edge detection again on large contours
#     updated_contours, updated_hierarchy = process_large_contours(image, contours, hierarchy, area_threshold=[10, 40, 500, 3000, 10000])
#     contours = updated_contours
#     hierarchy = updated_hierarchy

#     # Make copies of image for illustration
#     contour_image = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)
#     child_contour_image = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)
#     filled_image = image.copy()
#     mask_outer_combined = np.zeros_like(image)  # Single-channel mask
#     mask_inner_maxChid = np.zeros_like(image)  # Single-channel mask
#     cmap = cm.get_cmap('inferno', len(contours))
#     norm = Normalize(vmin=0, vmax=len(contours))

#     # Close the gaps between updated contours 
#     # for c, contour in enumerate(contours):
#     #     cv2.drawContours(contours2_image, contour, -1, 255, -1)
#     # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
#     # dilation = cv2.morphologyEx(contours2_image, cv2.MORPH_CLOSE, kernel, iterations=3)
#     # contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#     # hierarchy = hierarchy[0]

#     # Find parent contours - outer membrane - if there are children and area matches cell area
#     for c, contour in enumerate(contours):
        
#         # if hierarchy[c][2] and cv2.contourArea(contour) < 10000:      
#         if cv2.contourArea(contour) < 1000:    
#             cv2.drawContours(contour_image, [contour], -1, 255, -1)

#             color = cmap(norm(c))  # Get the color for the current contour
#             color_rgb = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))  # Convert to RGB
#             cv2.drawContours(child_contour_image, [contour], -1, color_rgb, 1)
#             # cv2.putText(child_contour_image, str(c), tuple(contour[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color_rgb, 1)
            
#             # Find child with largest area
#             child_idx = hierarchy[c][2]
#             max_area = 0
#             max_child = None
#             while child_idx != -1:
#                 child_hier = hierarchy[child_idx]  # next child in hierarchy
#                 area = cv2.contourArea(contours[child_idx]) # area of child                   
#                 if area > max_area:
#                     max_area = area
#                     max_child = child_idx
#                 max_area = max(area, max_area)
#                 child_idx = child_hier[0] 

#             # Fill the contour with the largest area 
#             if max_child is not None:
#                 # Create a mask for the inner contour
#                 cv2.drawContours(mask_inner_maxChid, [contours[max_child]], -1, 255, -1)

#             # Create a mask for the outer contour
#             # mask_outer = np.zeros_like(image)
#             # cv2.drawContours(mask_outer, [contour], -1, (255, 255, 255), -1)
#             cv2.drawContours(mask_outer_combined, [contour], -1, 255, -1)

#             # Compute the mean intensity within the area between the outer and inner contour
#             mean_intensity = cv2.mean(image, mask=mask_outer_combined)[0]

#             # Fill the contour on the filled image with the mean intensity
#             # filled_image[mask_inner_maxChid == 255] = mean_intensity

#             labels, markers = cv2.connectedComponents(filled_image)

#     _, axes = plt.subplots(nrows=2, ncols=4, figsize=(8, 8), sharex=True, sharey=True)
#     ax = axes.ravel()
#     ax[0].imshow(image, cmap=plt.cm.gray)
#     ax[0].set_title("Gaussian image")

#     ax[1].imshow(edges, cmap=plt.cm.gray)
#     ax[1].set_title("Edged image")

#     ax[2].imshow(dilation, cmap=plt.cm.gray)
#     ax[2].set_title("Edged after dilation")

#     ax[3].imshow(contour_image)
#     ax[3].set_title("Contour image")

#     ax[4].imshow(child_contour_image)
#     ax[4].set_title("Contour image with children in gradient colors")
#     sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#     sm.set_array([])  # Needed for ScalarMappable
#     plt.colorbar(sm, ax=ax[4], orientation='vertical', fraction=0.046, pad=0.04, label='Contour Index')

#     ax[5].imshow(mask_inner_maxChid)
#     ax[5].set_title("Inner mask max children")

#     ax[6].imshow(mask_outer_combined)
#     ax[6].set_title("Outer mask")

#     ax[7].imshow(filled_image, cmap=plt.cm.gray)
#     ax[7].set_title("Filled membranes")

#     return filled_image, labels, markers
    # %%
