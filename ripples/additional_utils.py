# from load_intan_rhd_format import read_data
import numpy as np
import pandas as pd
import csv

###Site Maps of the 4 shank-256 ch electrode arrays###

#First generation 256 ch array with 90 degree fold, no hook
siteMap_v1 = [123,124,121,122,119,120,117,118,115,116,113,114,110,112,109,111,144,143,141,139,128,129,130,131,132,133,134,135,136,137,138,146,145,142,140,108,107,148,147,106,105,150,149,104,103,152,151,102,101,154,153,125,126,127,155,156,99,100,157,158,97,98,159,160,
          96,95,162,161,94,93,164,163,92,91,166,165,90,89,168,167,88,87,170,169,86,85,172,171,84,83,179,177,174,173,
           181,182,183,184,185,186,191,189,187,188,190,180,178,176,175,80,82,79,81,77,78,75,76,73,74,71,72,69,70,67,
           68,65,66,64,63,61,62,59,60,57,58,55,56,53,54,51,52,49,50,46,48,45,47,208,207,205,203,193,195,196,194,192,
           197,198,199,200,201,202,210,209,206,204,44,43,212,211,42,41,214,213,40,39,216,215,38,37,218,217,36,35,220,
           219,34,33,222,221,31,32,224,223,29,30,225,226,27,28,227,228,0,1,2,230,229,26,25,232,231,24,23,234,233,22,
           21,236,235,20,19,243,241,238,237,245,246,247,248,249,250,251,252,253,254,255,244,242,240,239,16,18,15,17,
           13,14,11,12,9,10,7,8,5,6,3,4]

#Second generation 256 ch array with no fold, no hook
siteMap_v2 = [239,1,3,5,6,4,2,0,7,8,9,10,11,12,13,15,14,17,16,240,242,244,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,34,33,
              36,35,38,37,40,39,42,41,44,43,63,61,62,59,60,57,58,55,56,53,54,51,52,49,50,46,48,45,47,127,64,65,67,68,66,125,126,
              69,70,71,72,73,123,124,74,75,76,77,78,121,122,79,80,81,82,83,119,120,84,85,86,87,88,117,118,89,90,91,92,93,115,116,
              94,95,96,98,97,113,114,100,102,104,106,108,110,112,99,101,103,105,107,109,111,246,248,249,237,235,236,233,234,231,232,
              229,230,227,228,225,226,224,223,222,221,220,219,218,217,216,215,214,213,212,211,210,209,208,207,175,176,173,174,171,172,
              169,170,167,168,165,166,163,164,161,162,160,159,158,157,156,154,152,150,148,146,144,155,153,151,149,147,145,143,238,206,205,178,177,142,
              141,241,204,203,180,179,140,139,243,202,201,182,181,138,137,245,200,199,184,183,136,135,247,198,197,186,185,134,133,
              250,196,195,188,187,132,131,251,252,193,190,189,130,129,194,253,254,255,192,191,128]

#Third generation 256 ch array with no fold, hook
siteMap_v3 = [28,27,29,26,30,25,31,24,32,23,34,22,33,21,36,20,35,19,38,18,37,244,40,242,39,240,42,16,41,17,44,14,43,15,63,13,61,
              12,62,11,59,10,60,9,57,8,58,7,55,0,56,2,53,4,54,6,51,5,52,3,49,1,50,239,120,119,84,83,85,82,86,81,87,80,88,79,117,
              122,118,121,89,78,90,77,91,76,92,75,93,74,115,124,116,123,94,73,95,72,96,71,98,70,97,69,113,126,114,125,100,66,102,
              68,104,67,106,65,108,64,110,127,112,47,99,45,101,48,103,46,212,213,211,214,210,215,209,216,208,217,207,218,175,219,
              176,220,173,221,174,222,171,223,172,224,169,226,170,225,167,228,168,227,165,230,166,229,163,232,164,231,161,234,162,
              233,160,236,159,235,158,237,157,249,156,248,154,246,152,111,150,109,148,107,146,105,184,199,183,200,136,245,135,137,
              247,138,198,181,197,182,186,201,185,202,134,243,133,139,250,140,196,179,195,180,188,203,187,204,132,241,131,141,251,
              142,252,177,193,178,190,205,189,206,130,238,129,143,194,145,253,147,254,149,255,151,192,153,191,155,128,144]

def read_amplifier_dat_file(filepath):
    """
    This function reads the data from a .dat file created by Intan software and returns as a numpy array.

    Inputs:
        filepath: The path to the .dat file to be read.

    Outputs:
        amplifier_file: numpy array containing the data from the .dat file (in uV)
    """

    with open(filepath, 'rb') as fid:
        raw_array = np.fromfile(fid, np.int16)
    amplifier_file = raw_array * 0.195 #converting from int16 to microvolts
    return amplifier_file

def read_time_dat_file(filepath, sample_rate):
    """
    This function reads the time array from a time.dat file created by Intan software for a recording session,
    and returns the time as a numpy array.

    Inputs:
        filepath: The path to the time.dat file to be read.
        sample_rate: Sampling rate for the recording of interest.

    Outputs:
        time_file: Numpy array containing the time array (in s)
    """

    with open(filepath, 'rb') as fid:
        raw_array = np.fromfile(fid, np.int32)
    time_file = raw_array / float(sample_rate) #converting from int32 to seconds
    return time_file

def merge_amplifier_dat_files(folder, sr, missing_channels, port, num_channels):
    """This function is used for taking a recording session where the data is saved in the "one-file-per-channel" format of the Intan RHD/RHX software, and exporting it into a single .dat file that is in the "one-file-per-recording" format.

    Inputs:
        -folder (string): Directory of the folder containing the .dat files
        -sr (int): Sampling rate in Hz
        -missing_channels (1XN array with the N being the number of disabled channels): The list containing the amplifier indices of the channels that were disabled during the recording.
        -port (string): Intan amplifier port of the recording sites ('A' to 'D')
        -num_channels (int): Number of channels in the electrode array
    """

    time = read_time_dat_file(folder+'/time.dat',sr)

    #Generating the .dat files filled with zeros for the disabled channels
    for i in missing_channels:
        if i < 10:
            ch_prefix = '00'
        else:
            ch_prefix = '0'
        with open(folder+'/amp-'+port+'-'+ch_prefix+str(i)+'.dat', 'wb') as fid:
            data = np.zeros(len(time))
            data = data.astype('int16')
            fid.write(data)
            fid.close()

    #reading the data from the individual .dat files into a memmap array
    data = read_amplifier_dat_file(folder+'/amp-A-000.dat')
    data_all = np.memmap(folder + '/amplifier_data_temp.dat',dtype='int16', mode='w+', shape=(num_channels,len(data)))
    for i in range(num_channels):
        if i < 10:
            prefix = '00'
        else:
            prefix = '0'
        electrode_path = folder + '/amp-'+port+'-' + prefix + str(i) + '.dat'
        data_all[i] = read_amplifier_dat_file(electrode_path)

    #writing the data in the memmap array into a .dat file and deleting the temporary memmap array
    data_all = data_all.flatten('F')
    data_all.tofile(open(folder + '/amplifier_data.dat', 'wb'))
    os.remove(folder + '/amplifier_data_temp.dat')

    #deleting the individual .dat files in the folder
    for i in range(num_channels):
        if i < 10:
            prefix = '00'
        else:
            prefix = '0'
        electrode_path = folder + '/amp-'+port+'-' + prefix + str(i) + '.dat'
        os.remove(electrode_path)

def read_impedances_from_metadata_into_csv(folder, siteMap=siteMap_v3):
    """This function reads the impedance measurements stored in the Intan meta data and exports it into a .csv file, with channels ordered according to their spatial organization. This should be used in case the impedance measurements are not stored in a .csv file.

    Inputs:
        -folder (string): Directory to the folder containing the impedance measurements
        -siteMap (1xN list where N is the number of channels in the electrode array):
        the list that contains the information on how the Intan amplifier channels (e.g. 0-63 for port A-000 to A-063; 64-127 for port B-000 to B-063 etc.) are organized spatially (i.e. 0-63 for sites on first shank with 64 channels dorsal to ventral, 64-128 for sites on the second shank with 64 channels dorsal to ventral etc.)
    """

    meta_data = read_data(folder+'/info.rhd')
    amplifier_channels = meta_data['amplifier_channels']

    channel_ids = []
    impedances = []
    phases = []
    native_orders = []

    #siteMap = [27,28,12,29,11,20,7,21,10,1,15,8,26,18,31,23,9,2,14,6,25,17,30,22,0,19,13,24,4,16,3,5,43,32,56,45,60,35,52,40,41,33,46,38,57,49,62,54,42,34,47,39,58,50,63,55,59,44,61,36,53,37,51,48]

    for i in range(len(amplifier_channels)):
        channel_ids.append(amplifier_channels[i]['native_channel_name'])
        impedances.append(amplifier_channels[i]['electrode_impedance_magnitude'])
        phases.append(amplifier_channels[i]['electrode_impedance_phase'])
        native_orders.append(amplifier_channels[i]['native_order'])

    impedances_full = []
    phases_full = []
    count = 0
    for site in siteMap:
        if site in native_orders:
            impedances_full.append(impedances[count])
            phases_full.append(phases[count])
            count = count + 1
        else:
            impedances_full.append(10000000.0)
            phases_full.append(0)

    channel_info = list(zip(siteMap,impedances_full,phases_full))
    with open (folder+'/channel_impedances.csv','w',newline = '') as csvfile:
        my_writer = csv.writer(csvfile, delimiter = ',')
        my_writer.writerows(channel_info)

def order_impedance_measurements_in_csv(folder, siteMap=siteMap_v3):
    """This function reads the impedance measurements from the .csv file saved by
    the Intan RHX software, puts these measurements in spatial order of the channels,
    and saves the measurements in a new .csv file in this order, in the same folder
    where the original measurements were saved.
    Inputs:
        -folder (string): Directory to the folder containing the impedance measurements
        -siteMap (1xN list where N is the number of channels in the electrode array):
        the list that contains the information on how the Intan amplifier channels (e.g. 0-63 for port A-000 to A-063; 64-127 for port B-000 to B-063 etc.) are organized spatially (i.e. 0-63 for sites on first shank with 64 channels dorsal to ventral, 64-128 for sites on the second shank with 64 channels dorsal to ventral etc.)
     """

    #Data in the csv file is read into a Pandas dataframe
    df = pd.read_csv(folder + '/impedance.csv')
    impedances = df['Impedance Magnitude at 1000 Hz (ohms)']
    phases = df['Impedance Phase at 1000 Hz (degrees)']

    impedances_full = []
    phases_full = []
    channel_names = []

    #Going over the sites of the array in spatial order, and adding them to the output .csv file line by line,
    for site,i in enumerate(siteMap):
        impedances_full.append(impedances[i])
        phases_full.append(phases[i])
        channel_names.append(df['Channel Name'][i])

    #Writing the ordered impedance measurements into the output file.
    channel_info = list(zip(channel_names,impedances_full,phases_full))
    with open (folder+'/channel_impedances.csv','w',newline = '') as csvfile:
        my_writer = csv.writer(csvfile, delimiter = ',')
        my_writer.writerows(channel_info)
