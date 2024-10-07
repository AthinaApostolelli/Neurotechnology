function mean_channel_power = getChannelPower(window_timestamps, RawData, selected_channels, fband, fs);

% Compute the mean power of all channels for different frequency bands 
%
% INPUTS
% - window_timestamps        timestamps (s) of windows used for CC
% - RawData                  raw ephys data 
% - selected_channels        channel of interest
% - fband                    frequency band of selected signal 
% - fs                       ephys sampling rate
%
% OUTPUT
% mean_normalized_corr:    N x N matrix [N channels]
%
% HISTORY
% Athina Apostolelli 2024

num_windows = length(window_timestamps);
mean_channel_power = {zeros(length(selected_channels), 1), zeros(length(selected_channels), 1)};

selected_time = arrayfun(@(idx) round([window_timestamps(idx,1)*fs:window_timestamps(idx,2)*fs]), 1:num_windows, 'UniformOutput',false);
channel_data_windowed = arrayfun(@(idx) double(RawData(selected_channels, selected_time{idx})), 1:num_windows, 'UniformOutput',false);

channel_data_windowed_filt = arrayfun(@(idx) bandpass(channel_data_windowed{idx}.', fband, fs).', 1:num_windows, 'UniformOutput',false);
channel_data_windowed_power = arrayfun(@(idx) sum(channel_data_windowed_filt{idx}.^2, 2) / size(channel_data_windowed_filt{idx},2), 1:num_windows, 'UniformOutput',false);
mean_channel_power = mean(cell2mat(channel_data_windowed_power),2);

end
