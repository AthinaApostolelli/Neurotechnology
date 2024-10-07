"""
Created on 1/3/2024
@author: Athina Apostolelli

The threshold for considering the channel based on impedance is set to 1500000 Ohm. 
"""

import os, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from datetime import date
import scipy.stats as st


def get_impedance(impedance_files, threshold):
    mean_impedance_A = []
    mean_impedance_B = []
    se_impedance_A = []
    se_impedance_B = []
    ci_impedance_A = []
    ci_impedance_B = []

    mean_phase_A = []
    mean_phase_B = []
    se_phase_A = []
    se_phase_B = []
    ci_phase_A = []
    ci_phase_B = []

    data = {}

    for session in impedance_files:
        df_A = pd.read_csv(session, usecols=lambda x: x.lower() in ['channel name', 'impedance magnitude at 1000 hz (ohms)', 'impedance phase at 1000 hz (degrees)'], skiprows = list(range(65,129)))
        df_B = pd.read_csv(session, usecols=lambda x: x.lower() in ['channel name', 'impedance magnitude at 1000 hz (ohms)', 'impedance phase at 1000 hz (degrees)'], skiprows = list(range(1,65)))
                    
        df_A.rename(columns={'Impedance Magnitude at 1000 Hz (ohms)': 'Impedance'}, inplace=True)
        df_B.rename(columns={'Impedance Magnitude at 1000 Hz (ohms)': 'Impedance'}, inplace=True)
        
        df_A.rename(columns={'Impedance Phase at 1000 Hz (degrees)': 'Phase'}, inplace=True)
        df_B.rename(columns={'Impedance Phase at 1000 Hz (degrees)': 'Phase'}, inplace=True)

        # Remove channels with high impedance 
        df_A = df_A[df_A['Impedance'] <= threshold]
        df_B = df_B[df_B['Impedance'] <= threshold]

        ## IMPEDANCE AMPLITUDE
        # Mean and standard deviation of session impedance
        mean_session_impedance_A = np.mean(np.array(df_A['Impedance']))
        mean_session_impedance_B = np.mean(np.array(df_B['Impedance']))

        # Standard error of the mean of impedance 
        se_session_impedance_A = np.std(np.array(df_A['Impedance']), ddof=1) / np.sqrt(np.size(np.array(df_A['Impedance'])))
        se_session_impedance_B = np.std(np.array(df_B['Impedance']), ddof=1) / np.sqrt(np.size(np.array(df_B['Impedance'])))

        # 95% confidence intervals
        ci_session_impedance_A = st.t.interval(confidence=0.95, df=np.size(np.array(df_A['Impedance']))-1, loc=mean_session_impedance_A, scale=st.sem(np.array(df_A['Impedance'])))
        ci_session_impedance_B = st.t.interval(confidence=0.95, df=np.size(np.array(df_B['Impedance']))-1, loc=mean_session_impedance_B, scale=st.sem(np.array(df_B['Impedance'])))
        
        mean_impedance_A.append(mean_session_impedance_A)
        mean_impedance_B.append(mean_session_impedance_B)
        se_impedance_A.append(se_session_impedance_A)
        se_impedance_B.append(se_session_impedance_B)
        ci_impedance_A.append(ci_session_impedance_A)
        ci_impedance_B.append(ci_session_impedance_B)

        ## IMPEDANCE PHASE
        # Mean and standard deviation of session impedance phase
        mean_session_phase_A = np.mean(np.array(df_A['Phase']))
        mean_session_phase_B = np.mean(np.array(df_B['Phase']))

        # Standard error of the mean 
        se_session_phase_A = np.std(np.array(df_A['Phase']), ddof=1) / np.sqrt(np.size(np.array(df_A['Phase'])))
        se_session_phase_B = np.std(np.array(df_B['Phase']), ddof=1) / np.sqrt(np.size(np.array(df_B['Phase'])))

        # 95% confidence intervals
        ci_session_phase_A = st.t.interval(confidence=0.95, df=np.size(np.array(df_A['Phase']))-1, loc=mean_session_phase_A, scale=st.sem(np.array(df_A['Phase'])))
        ci_session_phase_B = st.t.interval(confidence=0.95, df=np.size(np.array(df_B['Phase']))-1, loc=mean_session_phase_B, scale=st.sem(np.array(df_B['Phase'])))
        
        mean_phase_A.append(mean_session_phase_A)
        mean_phase_B.append(mean_session_phase_B)
        se_phase_A.append(se_session_phase_A)
        se_phase_B.append(se_session_phase_B)
        ci_phase_A.append(ci_session_phase_A)
        ci_phase_B.append(ci_session_phase_B)

        # Output
        data['mean_impedance_A'] = np.array(mean_impedance_A)
        data['mean_impedance_B'] = np.array(mean_impedance_B)
        data['se_impedance_A'] = np.array(se_impedance_A)
        data['se_impedance_B'] = np.array(se_impedance_B)
        data['ci_impedance_A'] = np.array(ci_impedance_A).T
        data['ci_impedance_B'] = np.array(ci_impedance_B).T
        data['mean_phase_A'] = np.array(mean_phase_A)
        data['mean_phase_B'] = np.array(mean_phase_B)
        data['se_phase_A'] = np.array(se_phase_A)
        data['se_phase_B'] = np.array(se_phase_B)
        data['ci_phase_A'] = np.array(ci_phase_A).T
        data['ci_phase_B'] = np.array(ci_phase_B).T

    return data


def plot_impedance(impedance_files, animal, output_dir, data):
    
    x_values = np.arange(0,len(impedance_files))
    high_impedance_limit = np.ceil(np.ceil(np.max([data['mean_impedance_A'],data['mean_impedance_B']]) + np.max([data['ci_impedance_A'][1],data['ci_impedance_B'][1]])) + 100000)
    # impedance_range = np.arange(0, 1100000, 200000)
    impedance_range = np.arange(0, high_impedance_limit, 200000)  

    phase_range_min = np.floor(np.min([data['mean_phase_A'],data['mean_phase_B']]) - np.abs(np.min([data['ci_phase_A'][0],data['ci_phase_B'][0]])))
    phase_range_max = np.ceil(np.max([data['mean_phase_A'],data['mean_phase_B']]) + np.abs(np.max([data['ci_phase_A'][1],data['ci_phase_B'][1]])))
    # phase_range_min = np.floor(np.min([data['mean_phase_A'] - data['se_phase_A'], data['mean_phase_B'] - data['se_phase_B']])) #+ high_impedance_limit  # shift by the max of the first 'plot'
    # phase_range_max = np.ceil(np.max([data['mean_phase_A'] + data['se_phase_A'], data['mean_phase_B'] + data['se_phase_B']])) #+ high_impedance_limit
    phase_range = np.arange(np.min([phase_range_min,phase_range_max]), np.max([phase_range_min,phase_range_max])+30, 20)
    # phase_range = np.arange(phase_range_min, phase_range_max+10, 20)
    
    # fig = plt.figure(figsize=(10,10))
    # plt.rcParams.update({'font.size': 12})
    # gs = gridspec.GridSpec(2, 1, height_ratios=[2,3]) 

    # PLOT IMPEDANCE
    fig, ax0 = plt.subplots(1, 1, figsize=(10, 6))
    plt.rcParams.update({'font.size': 12})
        
    # ax0 = plt.subplot(gs[1])
    
    # errorbar 
    ax0.errorbar(x_values, data['mean_impedance_A'], yerr=data['ci_impedance_A'], label='Right (A)', color='r', linewidth=2, elinewidth=1, capsize=5)
    ax0.errorbar(x_values, data['mean_impedance_B'], yerr=data['ci_impedance_B'], label='Left (B)', color='b', linewidth=2, elinewidth=1, capsize=5)
    
    # lines with shaded area
    # line0, = ax0.plot(x_values, data['mean_impedance_A'], label='Right (A)', color='r', linewidth=2)
    # line1, = ax0.plot(x_values, data['mean_impedance_B'], label='Left (B)', color='b', linewidth=2)
    # ax0.fill_between(x_values, data['mean_impedance_A'] - data['se_impedance_A'], data['mean_impedance_A'] + data['se_impedance_A'], alpha=0.3, color='r')
    # ax0.fill_between(x_values, data['mean_impedance_B'] - data['se_impedance_B'], data['mean_impedance_B'] + data['se_impedance_B'], alpha=0.3, color='b')
    
    ax0.set_xticks(x_values)
    ax0.set_xticklabels(x_values+1)
    ax0.set_yticks(impedance_range)  # in Ohm
    ax0.set_yticklabels(np.arange(impedance_range[0]/1000, (impedance_range[-1]+100000)/1000, 200))  # in kOhm
    ax0.legend(loc='upper right')
    ax0.set_ylim(impedance_range[0], impedance_range[-1]+100000)
    ax0.set_xlabel('Week')
    ax0.set_ylabel('Impedance (k' + r'$\Omega$' +')')
    ax0.yaxis.set_label_coords(-.1, .5)

    # Save impedance plot
    plt.rcParams['svg.fonttype'] = 'none'
    plt.savefig(os.path.join(output_dir, 'impedance_' + animal + '_' + str(date.today()) + '.png'))
    plt.savefig(os.path.join(output_dir, 'impedance_' + animal + '_' + str(date.today()) + '.svg'), format='svg')


    # PLOT PHASE
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    plt.rcParams.update({'font.size': 12})

    # ax1 = plt.subplot(gs[0], sharex = ax0)

    # errorbar 
    ax1.errorbar(x_values, data['mean_phase_A'], yerr=data['ci_phase_A'], label='Right (A)', color='r', linewidth=2, elinewidth=1, capsize=5)
    ax1.errorbar(x_values, data['mean_phase_B'], yerr=data['ci_phase_B'], label='Left (B)', color='b', linewidth=2, elinewidth=1, capsize=5)
    
    # lines with shaded area
    # line2, = ax1.plot(x_values, data['mean_phase_A'], label='Right (A)', color='r', linewidth=2)
    # line3, = ax1.plot(x_values, data['mean_phase_B'], label='Left (B)', color='b', linewidth=2)
    # ax1.fill_between(x_values, data['mean_phase_A'] - data['se_phase_A'], data['mean_phase_A'] + data['se_phase_A'], alpha=0.3, color='r')
    # ax1.fill_between(x_values, data['mean_phase_B'] - data['se_phase_B'], data['mean_phase_B'] + data['se_phase_B'], alpha=0.3, color='b')
    ax1.set_xticks(x_values)
    ax1.set_xticklabels(x_values+1)
    ax1.legend(loc='upper right')
    ax1.set_ylim(phase_range[0]-10, phase_range[-1]+10)
    ax1.set_yticks(phase_range)
    ax1.set_ylabel('Phase (degrees)')
    ax1.set_xlabel('Week')
    ax1.yaxis.set_label_coords(-.1, .5)

    # Merge the two subplots
    # ax1.spines['bottom'].set_color('silver')
    # ax1.tick_params(axis='x', colors='silver')
    # plt.setp(ax1.get_xticklabels(), visible=False)
    # plt.subplots_adjust(hspace=.0)
   
    # Save figures
    plt.rcParams['svg.fonttype'] = 'none'
    plt.savefig(os.path.join(output_dir, 'phase_' + animal + '_' + str(date.today()) + '.png'))
    plt.savefig(os.path.join(output_dir, 'phase_' + animal + '_' + str(date.today()) + '.svg'), format='svg')


def custom_sorted_key(filename):
    first_number = ''.join(filter(str.isdigit, os.path.basename(filename)))
    return int(first_number)

if __name__ == "__main__":

    # Define parameters
    animals = ['rEO_05', 'rEO_06']
    # animals = ['rEO_06']
    basepath = 'D:\Rat_Recording'
    threshold = 1500000
    output_dir = os.path.join(basepath, 'eminhan_impedances')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Import impedance 
    for animal in animals:
        input_dir = os.path.join(basepath, animal)
        
        if animal == 'rEO_06':
            impedance_files = glob.glob(os.path.join(input_dir, '*_impedance.csv'))
            impedance_files = sorted(impedance_files, key=custom_sorted_key)
            impedance_files.insert(5, os.path.join(input_dir, '6_impedances_eis', '1khz_a-right_b-left.csv')) # session 6 
            
            impedance_files_to_keep = []
            for f, file in enumerate(impedance_files):
                if ('11' not in os.path.basename(file)) and ('12' not in os.path.basename(file)):
                    impedance_files_to_keep.append(file)
            print(impedance_files_to_keep)
            
            data = get_impedance(impedance_files_to_keep, threshold)

            plot_impedance(impedance_files_to_keep, animal, output_dir, data)

        elif animal == 'rEO_05':
            impedance_files = [r'D:\Rat_Recording\rEO_05\session-1_230522_161409\Aright-Bleft-post-recovery-post-rec.csv', 
                            r'D:\Rat_Recording\rEO_05\session-2_230606_143306\Impedance-Aright-Bleft.csv', 
                            r'D:\Rat_Recording\rEO_05\session-3_230706_160753\session-3_impedances_Aright_Bleft_postrecording.csv', 
                            r'D:\Rat_Recording\rEO_05\session-4_230825_165102\session-4-Aright-Bleft.csv',
                            r'D:\Rat_Recording\rEO_05\session-5_231005_161414\Aright-Bleft-postop-051023.csv']
            
            data = get_impedance(impedance_files, threshold)

            plot_impedance(impedance_files, animal, output_dir, data)