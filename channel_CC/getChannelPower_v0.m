function mean_channel_power = getChannelPower(selected_channels, ripple_timestamps, shuffled_ripple_timestamps, theta_windows, mode, RawData, fs, f_ripple, f_theta);

% Athina Apostolelli 2024

num_ripples = length(ripple_timestamps);
num_windows = length(theta_windows);
mean_channel_power = {zeros(length(selected_channels), 1), zeros(length(selected_channels), 1)};
fband = {f_ripple, f_theta};

switch mode 
    case 'swr'
        selected_time = arrayfun(@(idx) round([ripple_timestamps(idx,1)*fs:ripple_timestamps(idx,2)*fs]), 1:num_ripples, 'UniformOutput',false);
        channel_data_windowed = arrayfun(@(idx) double(RawData(selected_channels, selected_time{idx})), 1:num_ripples, 'UniformOutput',false);
        for f = 1:length(fband)
            channel_data_windowed_filt = arrayfun(@(idx) bandpass(channel_data_windowed{idx}.', fband{f}, fs).', 1:num_ripples, 'UniformOutput',false);
            channel_data_windowed_power = arrayfun(@(idx) sum(channel_data_windowed_filt{idx}.^2, 2) / size(channel_data_windowed_filt{idx},2), 1:num_ripples, 'UniformOutput',false);
            mean_channel_power{f} = mean(cell2mat(channel_data_windowed_power),2);
        end

    case 'non_swr'
        selected_time = arrayfun(@(idx) round([shuffled_ripple_timestamps(idx,1)*fs:shuffled_ripple_timestamps(idx,2)*fs]), 1:num_ripples, 'UniformOutput',false);
        channel_data_windowed = arrayfun(@(idx) double(RawData(selected_channels, selected_time{idx})), 1:num_ripples, 'UniformOutput',false);
        for f = 1:length(fband)
            channel_data_windowed_filt = arrayfun(@(idx) bandpass(channel_data_windowed{idx}.', fband{f}, fs).', 1:num_ripples, 'UniformOutput',false);
            channel_data_windowed_power = arrayfun(@(idx) sum(channel_data_windowed_filt{idx}.^2, 2) / size(channel_data_windowed_filt{idx},2), 1:num_ripples, 'UniformOutput',false);
            mean_channel_power{f} = mean(cell2mat(channel_data_windowed_power),2);
        end

    case 'theta'
        % num_windows = 2;
        selected_time = arrayfun(@(idx) round([theta_windows(idx,1)*fs:theta_windows(idx,2)*fs]), 1:num_windows, 'UniformOutput',false);
        channel_data_windowed = arrayfun(@(idx) double(RawData(selected_channels, selected_time{idx})), 1:num_windows, 'UniformOutput',false);
        for f = 1:length(fband)
            channel_data_windowed_filt = arrayfun(@(idx) bandpass(channel_data_windowed{idx}.', fband{f}, fs).', 1:num_windows, 'UniformOutput',false);
            channel_data_windowed_power = arrayfun(@(idx) sum(channel_data_windowed_filt{idx}.^2, 2) / size(channel_data_windowed_filt{idx},2), 1:num_windows, 'UniformOutput',false);
            mean_channel_power{f} = mean(cell2mat(channel_data_windowed_power),2);
        end
end
end


