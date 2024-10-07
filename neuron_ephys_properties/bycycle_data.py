#%%
import numpy as np
import scipy, os
import matplotlib.pyplot as plt

from neurodsp.filt import filter_signal
from neurodsp.plts import plot_time_series

from bycycle import BycycleGroup, Bycycle
from bycycle.plts.features import plot_feature_hist
from bycycle.utils.download import load_bycycle_data

from neurodsp.filt import filter_signal
from neurodsp.plts import plot_time_series
from neurodsp.sim import sim_combined

from bycycle import Bycycle
from bycycle.cyclepoints import find_extrema, find_zerox
from bycycle.cyclepoints.zerox import find_flank_zerox
from bycycle.plts import plot_cyclepoints_array
from bycycle.utils.download import load_bycycle_data


# Load data
basepath = 'I:/Yanik Lab Dropbox/Peter Gombkoto/Localization Manuscript 2024/RAT DATA/rEO_06/6_240205_150311'
fs = 2000

channels = np.arange(0,128,1)
num_channels = len(channels)
new_file = np.memmap(os.path.join(basepath, 'amplifier.lfp'), dtype='int16', mode='r')

data = np.reshape(new_file, (num_channels, int(len(new_file[:])/num_channels)), order='F') 
sig = data[93] # str. pyr.


#%% Preprocess signal 
f_lowpass = 30
n_seconds_filter = 0.5

# Lowpass filter
sig_low = filter_signal(sig, fs, 'lowpass', f_lowpass,
                        n_seconds=n_seconds_filter, remove_edges=False)

# Plot signal
times = np.arange(0, len(sig)/fs, 1/fs)
xlim = (1318, 1319)
tidx = np.logical_and(times >= xlim[0], times < xlim[1])

plot_time_series(times[tidx], [sig[tidx], sig_low[tidx]], colors=['k', 'k'], alpha=[.5, 1], lw=2)
# plt.show()

#%% Compyte cycle-by-cycle features 
f_range = (6,10) # theta
n_seconds_theta = .75

sig_narrow = filter_signal(sig, fs, 'bandpass', f_range,
                           n_seconds=n_seconds_theta, remove_edges=False)

plot_time_series(times[tidx], sig_narrow[tidx], xlabel="Time (s)", ylabel="CA1 Voltage (uV)")
# plt.show()

# Find rising and falling zerocrossings (narrowband)
rise_xs = find_flank_zerox(sig_narrow, 'rise')
decay_xs = find_flank_zerox(sig_narrow, 'decay')

peaks, troughs = find_extrema(sig_narrow, fs, f_range,
                              filter_kwargs={'n_seconds':n_seconds_theta})

plot_cyclepoints_array(sig_narrow, fs, peaks=peaks, troughs=troughs, xlim=xlim)
# plt.show()

#%% Compute features
# Fit
thresholds = {
    'amp_fraction_threshold': .3,
    'amp_consistency_threshold': .4,
    'period_consistency_threshold': .5,
    'monotonicity_threshold': .6,
    'min_n_cycles': 3
}

narrowband_kwargs = {'n_seconds': .5}

bm = Bycycle(
    center_extrema='trough',
    burst_method='cycles',
    thresholds=thresholds,
    find_extrema_kwargs={'filter_kwargs': narrowband_kwargs}
)

bm.fit(sig, fs, f_range)

df_ca1 = bm.df_features

# Limit analysis only to oscillatory bursts
# Each row of this table corresponds to an individual segment of the signal, 
# or a putative cycle of the rhythm of interest.
ca1_cycles = df_ca1[df_ca1['is_burst']]

#%% 
putative_windows = []

win_lims = np.where(np.diff(ca1_cycles.index) != 1)[0]
c = 0
for i in range(len(win_lims)):
    if i < len(win_lims):
        xlim = (ca1_cycles.sample_last_peak[ca1_cycles.index[c]], ca1_cycles.sample_last_peak[ca1_cycles.index[win_lims[i]]])
        c = win_lims[i] + 1
    else:
        xlim = (ca1_cycles.sample_last_peak[ca1_cycles.index[c]], ca1_cycles.sample_last_peak[ca1_cycles.index[-1]])
    ca1_plt = sig_narrow[xlim[0]:xlim[1]]
    times = np.arange(0, len(sig)/fs, 1/fs)
    tidx = np.logical_and(times >= xlim[0]/fs, times < xlim[1]/fs)   

    # fig, axes = plt.subplots(figsize=(15, 6), nrows=1)
    # plot_time_series(times[tidx], ca1_plt, xlabel="Time (s)", ylabel="CA1 Voltage (uV)") 
    # plt.show()

    # Save for matlab 
    putative_windows.append(np.array(xlim))

putative_windows_sec = [(round(s1/fs), round(s2/fs)) for s1, s2 in putative_windows]

# Merge consecutive windows
merged_windows = [putative_windows_sec[0]]
for current_win in putative_windows_sec[1:]:
    last_win = merged_windows[-1]
    
    if last_win[1] + 1 >= current_win[0]:
        merged_windows[-1] = (last_win[0], max(last_win[1], current_win[1]))
    else:
        merged_windows.append(current_win)
        
# Duration threshold
remove_idx = []
for w, win in enumerate(merged_windows):
    if win[1] - win[0] < 3:
        remove_idx.append(w)

theta_windows = [win for w, win in enumerate(merged_windows) if w not in remove_idx]

scipy.io.savemat(os.path.join(basepath, os.path.basename(basepath) + ".bycycle_theta_windows.mat"), {"theta_windows": theta_windows})
print("Done!")
    

# xlim = (ca1_cycles.index[0], ca1_cycles.index[20])
# ca1_plt = sig[xlim[0]:xlim[1]]
# times = np.arange(0, len(sig), 1)
# tidx = np.logical_and(times >= xlim[0], times < xlim[1])

# fig, axes = plt.subplots(figsize=(15, 6), nrows=1)

# plot_time_series(times[tidx], ca1_plt, ax=axes[0], xlabel="Time (s)", ylabel="CA1 Voltage (uV)")
# bm.plot(xlim=times[tidx])
# plt.show()

