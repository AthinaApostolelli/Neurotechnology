"""
author: Athina Apostolelli
adapted from: Tansel Baran Yasar and Peter Gombkoto

This script is used for filtering and downsampling certain channels of the raw multi-channel electrophysiology 
data stored in a binary .dat file with int16 format, and storing the downsampled LFP data in another binary .dat 
file also with int16 format. The script assumes the name of the data file is amplifier.dat.

NOTE that there is a memory limit that has not been addressed yet. If loading the entire .dat file, 
the program will likely crash depending on the number of electrode channels and the duration of the recording.
"""

import numpy as np
from scipy.signal import filtfilt, butter, decimate
import sys
import os
import matplotlib.pyplot as plt


def import_lfp(raw_data_file, num_channels, channels, sample_rate, verbose=False):
    """
    INPUTS
    - raw_data_file           file with raw recording (int16)
    - num_channels            number of channels in raw file 
    - channels                channels to load 
    - sample_rate             sampling rate of raw recording
    - verbose                 whether to display data indices while loading
    """

    bit_to_uV = 0.195
    chunk_size = 50 * sample_rate
    
    # Import .dat faster 
    new_file = np.memmap(raw_data_file, dtype='int16', mode='r')

    # Load the data in the data list
    data = [[] for _ in channels]

    for i, channel in enumerate(channels):
        for j in range(channel, len(new_file[:]), num_channels*chunk_size):

            if (j + num_channels*chunk_size) > len(new_file[:]):
                if verbose is True:
                    print("Processing data indices from %d to %d"%(j, len(new_file[:])))
                data[i].extend(new_file[j : len(new_file[:]) : num_channels])
            else:
                if verbose is True:
                    print("Processing data indices from %d to %d"%(j, j + chunk_size*num_channels))
                data[i].extend(new_file[j : (j + chunk_size*num_channels) : num_channels])

    # Convert to numpy array 
    data = np.array(data) 
    data = np.reshape(data, (len(channels), len(data[0]))) * bit_to_uV

    return data

########### SET PARAMETERS HERE #################
if __name__ == "__main__":

    sample_rate = 20000
    ds_sample_rate = 2000
    ds_factor = int(sample_rate / ds_sample_rate)
    num_channels = 128
    channels = [89,102,88,103,90,101,98,95,36,26,29,15,35,46,47,49] # channels are 0-indexed so last channel is 127 (0-63 = RIGHT, 64-127 = LEFT)
    chunk_size = 50 * sample_rate
      
    animal = 'rEO_07'
    sessname = 'session_1_2_230904_173347'  
    raw_data_dir = os.path.join('D:/Rat_Recording', animal, sessname)
    
    ds_data_dir = 'C:/Users/RECORDING/Athina/Ripples/lfp_data/'
    output_dir = os.path.join(ds_data_dir, animal)
    bit_to_uV = 0.195

    if not os.path.exists(ds_data_dir):
        os.mkdir(ds_data_dir)
    
    if not os.path.exists(output_dir):
            os.mkdir(output_dir)
    ##############################################


    raw_data_file = os.path.join(raw_data_dir, 'amplifier.dat')
    ds_data_file = os.path.join(output_dir, sessname + '.amplifier_ds.lfp')
    # xml_file = output_dir + sessname + '.amplifier.xml'
    # nrs_file = output_dir + sessname + '.amplifier.nrs'
    # ds_data_file = output_dir + sessname + '.amplifier_ds.lfp'
    xml_file = output_dir + sessname + '.amplifier_ds.xml'
    nrs_file = output_dir + sessname + '.amplifier_ds.nrs'


    data = import_lfp(raw_data_file=raw_data_file, num_channels=num_channels, channels=channels, sample_rate=sample_rate, verbose=True)

    print("The shape of the data array is: ", data.shape)

    # Filter the data
    # 3rd order Butterworth bandpass filter for 1-300Hz 
    b, a = butter(3, [1,300], btype='bandpass', fs=20000)
    data_flt = filtfilt(b, a, data)

    # Optionally plot a figure of the filter
    # w, h = freqz(b, a, worN=8000)
    # plt.figure(figsize=(10, 6))
    # plt.plot(0.5 * 20000 * w / np.pi, np.abs(h), 'b')
    # plt.title('Butterworth Bandpass Filter Frequency Response')
    # plt.xlabel('Frequency [Hz]')
    # plt.ylabel('Gain')
    # plt.grid()
    # plt.show()

    # Downsample the data    
    data_ds = decimate(data_flt, ds_factor)

    # Save to file in format supported by Neuroscope
    data_ds = data_ds.flatten('F')
    data_ds = data_ds.astype('int16')

    if os.path.exists(ds_data_file):
        os.remove(ds_data_file)
    if os.path.exists(xml_file):
        os.remove(xml_file)
    if os.path.exists(nrs_file):
        os.remove(nrs_file)

    f = open(ds_data_file,'wb')
    data_ds.tofile(f)
    f.close()
    print("Done!")