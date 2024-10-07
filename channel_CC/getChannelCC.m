function mean_normalized_corr = getChannelCC(window_timestamps, RawData, meanData, selected_channels, fband, fs)

% Compute the cross-correlation (CC) among all channels  
%
% INPUTS
% - window_timestamps        timestamps (s) of windows used for CC
% - RawData                  raw ephys data
% - meanData                 mean noise levels 
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

selected_time = cell(num_windows,1);
data_windowed = cell(num_windows,1);
data_windowed_filt = cell(num_windows,1);
data_windowed_power = cell(num_windows,1);
ch_corr = cell(num_windows,1);
normalized_corr = cell(num_windows,1);
auto_corr = cell(num_windows,1);

for i = 1:num_windows
    selected_time{i} = round([window_timestamps(i,1)*fs:window_timestamps(i,2)*fs]);
    if ~isempty(meanData)
        data_windowed{i} = double(RawData(selected_channels,selected_time{i})) - meanData(selected_time{i});
    else
        data_windowed{i} = double(RawData(selected_channels,selected_time{i}));
    end
    data_windowed_filt{i} = bandpass(data_windowed{i}.', fband, fs).';
    ch_corr{i} = data_windowed_filt{i}*data_windowed_filt{i}.';
    
    % min-max normalization for each time window 
    min_auto = min(ch_corr{i}(:));
    max_auto = max(ch_corr{i}(:));
    ch_corr{i} = rescale(ch_corr{i},-1,1,'InputMin',min_auto,'InputMax',max_auto);
    
    normalized_corr{i} = ch_corr{i};
    temp_data = normalized_corr{i};
    temp_data(isnan(temp_data)) = 0;
    normalized_corr{i} = temp_data;
    auto_corr{i} = diag(diag(normalized_corr{i}));
end

% Mean cross-correlation across all windows
concat_normalized_corr = cat(3,normalized_corr{:});
mean_normalized_corr = mean(concat_normalized_corr, 3);  

end