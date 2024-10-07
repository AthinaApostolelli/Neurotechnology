"""
author: Athina Apostolelli
Created on 23/1/2024

This script is used for detecting the time shift between SWRs from two hemispheres. 
"""

import os
import sys
import scipy
import math 
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import filtfilt, butter, hilbert, detrend

from downsample_filter_LFP import import_lfp

def sort_events(ripple_timestamps, ripple_classes, num_ripples, lfp_detrend=None, method='start'):
    
    if method == 'start':
        # Events are sorted according to their start time
        sorted_indices = np.argsort(np.array(ripple_timestamps)[:,0])
        sorted_timestamps = np.array(ripple_timestamps)[sorted_indices].tolist()
        sorted_classes = np.array(ripple_classes)[sorted_indices].tolist()
        sorted_centers = [(sorted_timestamps[r][0] + sorted_timestamps[r][1]) / 2 for r in range(num_ripples)]

    elif method == 'min':
        # Events are sorted according to when the trough occurs
        if lfp_detrend is None:
            sys.exist('To sort the SWRs according to the LFP trough, please provide the LFP data.')

        lfp_trough_idx = []
        for r in range(num_ripples):
            window = np.arange(math.floor(ripple_timestamps[r][0] * 2000), math.floor(ripple_timestamps[r][1] * 2000))
            if ripple_classes[r] == 'left':
                lfp_trough_idx.append(np.where(lfp_detrend[0:8,window] == np.min(lfp_detrend[0:8,window]))[0] + window[0])
            elif ripple_classes[r] == 'right':
                lfp_trough_idx.append(np.where(lfp_detrend[8:16,window] == np.min(lfp_detrend[8:16,window]))[0] + window[0])

        sorted_indices = np.argsort(np.array(lfp_trough_idx).flatten())
        sorted_timestamps = np.array(ripple_timestamps)[sorted_indices].tolist()
        sorted_classes = np.array(ripple_classes)[sorted_indices].tolist()
        sorted_centers = [(sorted_timestamps[r][0] + sorted_timestamps[r][1]) / 2 for r in range(num_ripples)]

    elif method == 'max':
        # Events are sorted according to when the peak occurs
        if lfp_detrend is None:
            sys.exist('To sort the SWRs according to the LFP peak, please provide the LFP data.')

        lfp_peak_idx = []
        for r in range(num_ripples):
            window = np.arange(math.floor(ripple_timestamps[r][0] * 2000), math.floor(ripple_timestamps[r][1] * 2000))  # TODO: add +1 to the window?
            if ripple_classes[r] == 'left':
                lfp_peak_idx.append(np.where(lfp_detrend[0:8,window] == np.max(lfp_detrend[0:8,window]))[0] + window[0])
            elif ripple_classes[r] == 'right':
                lfp_peak_idx.append(np.where(lfp_detrend[8:16,window] == np.max(lfp_detrend[8:16,window]))[0] + window[0])

        sorted_indices = np.argsort(np.array(lfp_peak_idx).flatten())
        sorted_timestamps = np.array(ripple_timestamps)[sorted_indices].tolist()
        sorted_classes = np.array(ripple_classes)[sorted_indices].tolist()
        sorted_centers = [(sorted_timestamps[r][0] + sorted_timestamps[r][1]) / 2 for r in range(num_ripples)]

    return sorted_timestamps, sorted_centers, sorted_classes

def get_event_overlaps(sorted_timestamps, sorted_centers, num_ripples):
    overlaps = []
    for r in range(0,num_ripples-1):
        if sorted_timestamps[r+1][0] <= sorted_centers[r] <= sorted_timestamps[r+1][1]:  
            overlaps.append([r,r+1])
            r += 2
    print('Fraction of overlapping SWRs: %d%%' %(len(overlaps)*2/num_ripples*100))

    return overlaps 

def get_timeshift_event_start(ripple_classes, ripple_timestamps, num_ripples, output_dir):
    """
    First find the SWRs that overlap in time between the two hemispheres. 
    Then calculate the time shift between the overlapping SWRs according to the start of each event. 
    """

    # Sort the SWRs according to the start time
    [sorted_timestamps, sorted_centers, sorted_classes] = sort_events(ripple_timestamps, ripple_classes, num_ripples, method='start')
    
    # Search for time overlaps (double counted)
    overlaps = get_event_overlaps(sorted_timestamps, sorted_centers, num_ripples)

    # Calculate the time shift in the SWR start between the two hemispheres 
    timeshift = []
    for p in overlaps:
        left_idx = 0 if sorted_classes[p[0]]=='left' else 1
        right_idx = 0 if sorted_classes[p[0]]=='right' else 1
        shift = sorted_timestamps[p[left_idx]][0] - sorted_timestamps[p[right_idx]][0]  ## Confirm this is correct 
        timeshift.append(shift)
    
    # Save time shifts 
    hemisphere_labels = []
    starts = []
    for p in overlaps:
        hemisphere_labels.append(np.array(sorted_classes)[p].tolist())
        starts.append([sorted_timestamps[p[0]][0], sorted_timestamps[p[1]][0]])

    output_array = []
    for p, pair in enumerate(overlaps):
        output_array.append([hemisphere_labels[p][0], hemisphere_labels[p][1], starts[p][0], starts[p][1], timeshift[p]])

    np.savetxt(output_dir + '/timeshifts.txt', output_array, fmt='%s', header=','.join(['hemisphere_1', 'hemisphere_2', 'start_1', 'start_2', 'time_shift']))
    
    # Plot histogram of time shifts 
    plt.close('all')
    plt.ioff()
    plt.hist(timeshift, bins=30)
    plt.title('Time shift of SWRs detected from left vs right hemisphere')
    plt.xlabel('Time shift [s]')
    plt.ylabel('Number of SWRs')
    plt.savefig(output_dir + '/event_start_timeshift_histogram.png')

def get_timeshift_peak_trough(sorted_timestamps, sorted_classes, lfp_detrend, num_ripples, output_dir):
    timeshift_peak_trough = [] 

    for r in range(num_ripples):
        lfp_index_start = math.floor(sorted_timestamps[r][0] * 2000)
        lfp_index_end = math.floor(sorted_timestamps[r][1] * 2000)
        window = np.arange(lfp_index_start, lfp_index_end)

        if sorted_classes[r]=='left':
            lfp_trough_idx = np.where(lfp_detrend[0:8,window] == np.min(lfp_detrend[0:8,window]))[0]
            lfp_peak_idx = np.where(lfp_detrend[0:8,window] == np.max(lfp_detrend[0:8,window]))[0]
        elif sorted_classes[r]=='right':
            lfp_trough_idx = np.where(lfp_detrend[8:16,window] == np.min(lfp_detrend[8:16,window]))[0]
            lfp_peak_idx = np.where(lfp_detrend[8:16,window] == np.min(lfp_detrend[8:16,window]))[0]

        shift = (lfp_peak_idx - lfp_trough_idx) / 2000 
        timeshift_peak_trough.append(shift[0])

    # Plot histogram of time shifts 
    plt.close('all')
    plt.ioff()
    plt.hist(timeshift_peak_trough, bins=15)
    plt.title('Difference in peak vs trough of LFP during a SWR')
    plt.xlabel('Time shift [s]')
    plt.ylabel('Number of SWRs')
    plt.savefig(output_dir + '/peak_trough_timeshift_histogram.png')

def get_timeshift_contra_peak_trough(sorted_timestamps, sorted_classes, overlaps, lfp_detrend, output_dir):
    timeshift_contra_peak_trough = []

    for p in overlaps:
        window0 = np.arange(math.floor(sorted_timestamps[p[0]][0] * 2000), math.floor(sorted_timestamps[p[0]][1] * 2000)) # 1st event in pair
        window1 = np.arange(math.floor(sorted_timestamps[p[1]][0] * 2000), math.floor(sorted_timestamps[p[1]][1] * 2000)) # 2nd event in pair 

        if sorted_classes[p[0]]=='left':
            lfp_trough_idx = np.where(lfp_detrend[0:8,window0] == np.min(lfp_detrend[0:8,window0]))[0]
            lfp_peak_idx = np.where(lfp_detrend[8:16,window1] == np.max(lfp_detrend[8:16,window1]))[0] # contralateral (right) peak 
        elif sorted_classes[p[0]]=='right':
            lfp_trough_idx = np.where(lfp_detrend[8:16,window0] == np.min(lfp_detrend[8:16,window0]))[0]
            lfp_peak_idx = np.where(lfp_detrend[0:8,window1] == np.max(lfp_detrend[0:8,window1]))[0] # contralateral (left) peak 
        
        shift = (lfp_peak_idx - lfp_trough_idx) / 2000 
        timeshift_contra_peak_trough.append(shift[0])

    # Save time shifts 
    hemisphere_labels = []
    for p in overlaps:
        hemisphere_labels.append(np.array(sorted_classes)[p].tolist())

    output_array = []
    for p in range(len(overlaps)):
        output_array.append([hemisphere_labels[p][0], hemisphere_labels[p][1], timeshift_contra_peak_trough[p]])

    np.savetxt(output_dir + '/timeshift_contra_peak_trough.txt', output_array, fmt='%s', header=','.join(['hemisphere_1', 'hemisphere_2', 'timeshift_contra_peak_trough']))

    # Plot histogram of time shifts 
    plt.close('all')
    plt.ioff()
    plt.figure(figsize=(8,6))
    plt.hist(timeshift_contra_peak_trough, bins=15)
    plt.title('Difference in peak vs contralateral trough of LFP during two overlapping SWRs')
    plt.xlabel('Time shift [s]')
    plt.ylabel('Number of SWRs')
    plt.savefig(output_dir + '/contra_peak_trough_timeshift_histogram.png')

if __name__ == "__main__":
    
    # Set parameters
    concatenated = True  
    ripples_dir = 'C:/Users/RECORDING/Athina/Ripples/' 
    animal = 'rEO_06/'
    
    if concatenated is False:
        sessname = '1_231218_153006'  
        curated_ripples_file = os.path.join(ripples_dir, 'curated_ripples', animal, sessname, 'curated_ripples.mat')
        lfp_file = os.path.join(ripples_dir, 'lfp_data', animal, sessname + '.amplifier_ds.lfp')

        output_dir = os.path.join(ripples_dir, 'timeshift_analysis', animal, sessname)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)  

        # Import manually curated SWRs
        ripple_data = scipy.io.loadmat(curated_ripples_file)
        ripple_classes = [label[0] for label in ripple_data['ripple_classes'].flatten()]
        ripple_timestamps = ripple_data['ripple_timestamps'].tolist()
        num_ripples = len(ripple_classes)

        # Import LFP (downsampled and filtered)
        lfp_data = np.array(import_lfp(raw_data_file=lfp_file, num_channels=16, channels=np.arange(0,16), sample_rate=2000, verbose=False))

        # Filter LFP for ripples and Hilbert transform (see Matlab function in LFPViewer_CNN_ripples)
        b, a = butter(600, [130,245], btype='bandpass', fs=2000)
        lfp_detrend = detrend(lfp_data, axis=1)
        lfp_filtered = filtfilt(b, a, lfp_detrend) # TODO: not working
        lfp_filtered_hilbert = np.array(abs(hilbert(lfp_filtered)))

        ######### Get timeshift (event start) - TODO: probably irrelevant #########
        get_timeshift_event_start(ripple_classes, ripple_timestamps, num_ripples, output_dir)

        ######### Get timeshift between peak of oriens and trough of radiatum in sharp wave #########
        # Sort the SWRs according to the start time
        [sorted_timestamps, sorted_centers, sorted_classes] = sort_events(ripple_timestamps, ripple_classes, num_ripples, lfp_detrend=lfp_detrend, method='min')

        # Search for time overlaps (double counted)
        overlaps = get_event_overlaps(sorted_timestamps, sorted_centers, num_ripples)

        # Shift between peak and trough of each event 
        get_timeshift_peak_trough(sorted_timestamps, sorted_classes, lfp_detrend, num_ripples, output_dir)

        # Shift between peak and trough of contralateral events 
        get_timeshift_contra_peak_trough(sorted_timestamps, sorted_classes, overlaps, lfp_detrend, output_dir)

        
    else:   
        sessions = [s for s in os.listdir(os.path.join(ripples_dir, 'curated_ripples', animal)) if os.path.isdir(os.path.join(ripples_dir, 'curated_ripples', animal, s))]
        output_dir = os.path.join(ripples_dir, 'timeshift_analysis', animal, 'concatenated_sessions')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)  
        
        # Import manually curated SWRs and timeshift data 
        concat_ripple_classes = []
        concat_ripple_timestamps = []
        concat_num_ripples = []

        timeshift_contra_peak_trough = []

        for sessname in sessions: 

            # Peak to contralateral trough timeshift 
            with open(os.path.join(ripples_dir, 'timeshift_analysis', animal, sessname, 'timeshift_contra_peak_trough.txt'), 'r') as file:
                lines = file.readlines()
            for line in lines[1::]:
                parts = line.split(' ')
                value = float(parts[-1])
                timeshift_contra_peak_trough.append(value)
            
        # Plot histogram of time shifts 
        plt.close('all')
        plt.ioff()
        plt.figure(figsize=(8,6))
        plt.hist(timeshift_contra_peak_trough, bins=15)
        plt.xlim([-0.004,0.004])
        plt.xticks(ticks=np.arange(-0.004,0.005,0.001), labels=np.arange(-4,5,1))
        plt.title('Difference in peak vs contralateral trough of LFP during two overlapping SWRs')
        plt.xlabel('Time shift [ms]')
        plt.ylabel('Number of SWRs')
        plt.savefig(output_dir + '/contra_peak_trough_timeshift_histogram.svg')
        plt.savefig(output_dir + '/contra_peak_trough_timeshift_histogram.png')



       

        