"""
author: Athina Apostolelli
adapted from: Valter Lundegardh and the rippl_AI repository functions

This script is used for detecting the SWRs from electrophysiology data that have been filtered and downsampled to either 1250 or 2000 Hz. 
The CNN developed by the Prida lab is used for SWR detection (https://github.com/PridaLab/rippl-AI).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
mplstyle.use('fast')
import os
import sys
import scipy
import h5py
import keras

rippl_AI_repo = 'C:/Users/RECORDING/Athina/Github/rippl-AI'
ripple_analysis_dir = 'C:/Users/RECORDING/Athina/Ripples/'
sys.path.insert(1, rippl_AI_repo)
import rippl_AI
import aux_fcn
from importlib import reload
reload(rippl_AI)
reload(aux_fcn)

from downsample_filter_LFP import import_lfp
from aux_fcn_nt import process_LFP_new


########### SET PARAMETERS AND LOAD DATA HERE ###########
if __name__ == "__main__":
    filtered = True # False if loading from the LFP.mat file (downsampled) or True if loading from the .lfp file (filtered and downsampled)
    comparison = False # whether to compare CNN-detected SWRs to manually curated ones
    animal = 'rEO_07'
    sessname = 'session_1_2_230904_173347' 
    downsampled_freqs = [2000]
    compare_structs = 'hemispheres' # 'regions'
    structures = ["left", "right"]
    threshold = 0.95
    arch = 'CNN1D'
    # training_batch = 64
    # epochs = 3000

    # Load retrained model 
    # model_dir = os.path.join(ripple_analysis_dir, 'retraining', 'explore_models', 'CNN1D_Ch8_Ts016_C00_E01_TB0064')
    ## model_dir = os.path.join(ripple_analysis_dir, 'retraining', 'model_' + arch + '_' + str(epochs) + 'epochs_' + str(training_batch) + 'bs_interpolation')
    # new_model = keras.models.load_model(model_dir, compile=False)

    # Comparison 
    if comparison is True: 
        comp_dir = os.path.join(ripple_analysis_dir, 'retraining', 'performance/' + animal + '/' + sessname + '/')
        if not os.path.exists(comp_dir):
            os.makedirs(comp_dir)

        # Import curated ripples
        if compare_structs == 'regions':
            ripples_dir = "C:/Users/RECORDING/Athina/Ripples/curated_ripples/" + animal + "/" + sessname + "/"
            curated_ripples_dHPC_file = "curated_ripples_dHP.mat"
            curated_ripples_iHPC_file = "curated_ripples_iHP.mat"
            
            with h5py.File(os.path.join(ripples_dir, curated_ripples_dHPC_file), 'r') as f:
                ripples_dHPC_times = np.transpose(f['ripple_timestamps'][:])  # n_ripples x 2 (start and end of ripple)
            
            with h5py.File(os.path.join(ripples_dir, curated_ripples_iHPC_file), 'r') as f:
                ripples_iHPC_times = np.transpose(f['ripple_timestamps'][:])  # n_ripples x 2 (start and end of ripple)
            
            true_events = [ripples_dHPC_times, ripples_iHPC_times]
            
            print("Curated dHPC ripples timestamps shape: ", ripples_dHPC_times.shape)
            print("Curated iHPC ripples timestamps shape: ", ripples_iHPC_times.shape)

        elif compare_structs == 'hemispheres':
            ripples_dir = "C:/Users/RECORDING/Athina/Ripples/curated_ripples/" + animal + "/" + sessname + "/"
            curated_ripples_file = "curated_ripples.mat"
            
            ripple_data = scipy.io.loadmat(os.path.join(ripples_dir, curated_ripples_file))
            ripple_classes = [label[0] for label in ripple_data['ripple_classes'].flatten()]
            ripple_timestamps = ripple_data['ripple_timestamps']
            
            left_indices = []
            left_indices.extend(i for i, label in enumerate(ripple_classes) if 'left' in label)
            
            right_indices = []
            right_indices.extend(i for i, label in enumerate(ripple_classes) if 'right' in label)

            ripples_left_times = ripple_timestamps[left_indices,:]  # n_ripples x 2 (start and end of ripple)
            ripples_right_times = ripple_timestamps[right_indices,:]  # n_ripples x 2 (start and end of ripple)
                    
            true_events = [ripples_left_times, ripples_right_times]
            print("Curated Left ripples timestamps shape: ", ripples_left_times.shape)
            print("Curated Right ripples timestamps shape: ", ripples_right_times.shape)

    # Load LFP 
    if filtered is False:
        lfp_dir = "I:/Dropbox (Yanik Lab)/Multiarea rat recordings/" + animal + "/preprocessed_data_35/" + sessname
        lfp_file = sessname + "_LFP.mat"  # this LFP file is already downsampled to 2000Hz
        
        with h5py.File(os.path.join(lfp_dir, lfp_file), 'r') as f:
            lfp = np.transpose(f['lfp']['data'][:])  # n_samples x n_channels 
    else:
        # lfp_dir = os.path.join(ripple_analysis_dir, 'lfp_data/', animal) 
        lfp_dir = 'F:/Rat_Recording/rEO_07/session_1_2_230904_173347'

        for file in os.listdir(lfp_dir):
            if '.lfp' in file:
            # if sessname in file and '.lfp' in file:
                lfp_file = os.path.join(lfp_dir, file)
                break  

    # Output dir
    output_dir = "C:/Users/RECORDING/Athina/Ripples/" + "detection/" + animal + "/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load data (n_channels x n_samples)
    print("Importing file: ", lfp_file)
    # lfp = import_lfp(raw_data_file=lfp_file, num_channels=16, channels=np.arange(0,16), sample_rate=2000, verbose=False)
    lfp = import_lfp(raw_data_file=lfp_file, num_channels=128, channels=[89,102,88,103,90,101,98,95,36,26,29,15,35,46,47,49], sample_rate=2000, verbose=True)
    lfp = np.transpose(lfp)

    ########################################
    
    # Fit the CNN1D model and get its predictions and performance 
    for f in downsampled_freqs:
        downsampled_fs = f
        
        spwrs_times_array = []

        for r, region in enumerate(structures):
            # Choose channels 
            if r == 0:
                # channels = [1,-1,-1,2,-1,-1,-1,4]
                channels = np.arange(0,8)
            else:
                # channels = [8,-1,-1,11,-1,-1,-1,13]
                channels = np.arange(8,16)

            # Predictions
            current_dir = os.getcwd()
            os.chdir(rippl_AI_repo)
            
            if downsampled_fs == 1250:
                SWR_prob, LFP_norm = rippl_AI.predict(lfp, sf=2000, arch='CNN1D', model_number=1, channels=channels)
            else:
                LFP_norm = process_LFP_new(lfp, sf=2000, downsampled_fs=downsampled_fs, channels=channels)
                # SWR_prob = aux_fcn.prediction_parser(LFP_norm, arch='CNN1D', new_model=new_model, n_channels=8, n_timesteps=16)
                SWR_prob = aux_fcn.prediction_parser(LFP_norm, arch='CNN1D', model_number=1, n_channels=8, n_timesteps=16)     
            
            os.chdir(current_dir)
            
            predictions = rippl_AI.get_intervals(y=SWR_prob, LFP_norm=LFP_norm, sf=downsampled_fs, win_size=30, threshold=threshold)  # TODO: close plots so that script runs to the end

            # plt.figure(figsize=(10, 6))
            # plt.plot(np.arange(4000,6000), SWR_prob[4000:6000])  # 1 s if sampling rate is 2000 Hz
            # plt.xlabel('Time [1 s]')
            # plt.ylabel('Probability')
            # plt.show()
            
            # Performance 
            if comparison is True:
                performance = aux_fcn.get_performance(pred_events=predictions, true_events=true_events[r], threshold=0)[0:3]
                with open(os.path.join(comp_dir, f"performance_CNN1D_{str(downsampled_fs)}Hz_{region}.txt"), 'w') as f:
                    f.write('\t'.join(['precision', 'recall', 'F1']) + '\n')
                    f.write('\t'.join(map(str, performance)) + '\n')

            # Merge ripples: the start of the second overlaps with the end of the first within the buffer time window
            buffer=0.0128  # seconds

            spwr_start = predictions[0, 0]
            spwr_end = predictions[0, 1]

            curr_start = predictions[0, 0]
            curr_end = predictions[0, 1]
            spwrs_times = []

            for i, (start_t, end_t) in enumerate(predictions[1:, :]):
                if (start_t <= curr_end+buffer): 
                    curr_end = end_t
                else:
                    spwr_end = curr_end
                    spwrs_times.append((spwr_start, spwr_end))
                    spwr_start = start_t
                    curr_end = end_t

            # Remove very short events
            for i, (spwr_start, spwr_end) in enumerate(spwrs_times):
                if (spwr_end - spwr_start) < 0.020:  # in s (20 ms)
                    spwrs_times.remove((spwr_start, spwr_end))

            spwrs_times_array.append(np.array(spwrs_times))

        # Output to .mat files
        all_spwrs_times = np.concatenate((spwrs_times_array[0], spwrs_times_array[1]))  # merge ripples from the two structures (dHPC and iHPC / left and right hemisphere)

        n_spwr1 = spwrs_times_array[0].shape[0]
        n_spwr2 = spwrs_times_array[1].shape[0]
        print("Number of %s spwrs: " %structures[0], n_spwr1)
        print("Number of %s spwrs: " %structures[1], n_spwr2)
        print("Total number spwrs: ", n_spwr1 + n_spwr2)

        list1 = [structures[0]]*n_spwr1
        list2 = [structures[1]]*n_spwr2

        labels = list1 + list2

        save_path = output_dir + str(downsampled_fs) + "Hz/" 
        if not os.path.exists(save_path):
            os.mkdir(save_path)
            
        scipy.io.savemat(save_path + sessname + ".ripples.mat"
                        , {"spwr": all_spwrs_times, "labels": np.transpose(np.asarray(labels, dtype='object')), "threshold": threshold})
        print("Done!")