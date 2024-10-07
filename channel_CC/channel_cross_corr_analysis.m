% This script calculates the cross-correlation (CC) among the recorded channels
% for specific frequency bands and time windows during the recording.
% Commonly, the recording is split between SWR times and non-SWR times and
% the power of the signal is used for the cross-correlation calculation. 
%
% OUTPUT
% channelCC struct
%   - (mode).(signal).corr  - min-max normalized CC value for (mode) time window and (signal) target signal defined by specific frequency band
%   - (mode).(signal).power - channel power for (mode) time window and (signal) target signal defined by specific frequency band
%   - (mode).fband          - frequency band to compute power and CC
%   - impedance             - impedance of selected channels
%   - params                - information on parameters of the script
% 
% HISTORY 
% Athina Apostolelli 2024
% Adapted from Eminhan Ozil 2023

%% Set parameters
basepath = 'D:\Rat_Recording\rEO_05';
sessname = '128ch_concatenated_sessions';
filename = fullfile(basepath, sessname, '128ch_concat_data.dat');
animal = 'rEO_05';

forceReload = true;                     % whether to recompute channelCC
loadImpedance = true;                   % whether to load impedance measurements
interpDeadChs = true;                   % whether to interpolate values for dead channels
fs = 20000;                             % ephys sampling rate (Hz)
f_ripple = [150,250];                   % high-frequency oscillation component of SWRs (Hz)
f_theta = [4,10];                       % theta frequency band (Hz)
theta_selected_channels = [36]+1;       % channel in hippocampus 

if ~forceReload & exist(fullfile(basepath, sessname, [sessname '.channelCC.mat']))
    disp('Channel cross-correlation already detected. Loading...');
    load(fullfile(basepath, sessname, [sessname '.channelCC.mat']));

else
    disp('Computing channel power and cross-correlation...');
    phys_ch_ord = [20 21 41 18 40 19 43 16 42 23 45 22 44 25 38 24 39 27 36 26 37 29 34 28 35 31 32 17 33 15 30 14 46 13 47 12 49 11 48 10 51 9 50 8 53 7 52 6 55 5 54 4 57 3 56 2 59 1 58 0 61 63 60 62]+1;
    % dead_chs = [20,21,22,25,24,28,31,30,14,13,4,63,62,50,8]+1; % concat data rEO_05
    dead_chs = [21,18,23,22,25,39,28,32,13]+1; % concat data rEO_06
    % dead_chs = [18,16,22,23,44,39,36,32,13,49,54,8,53,7,52,6]+1;  % right rEO_06
    % dead_chs = [99,96,97,76,74,72]+1;  % left rEO_06
    interp_chs = dead_chs;

    % Import raw data
    % NOTE: channels in raw data are not ordered
    nChansInRawFile = 128;
    bitScaling = 0.195;
    a = memmapfile(filename, 'Format','int16');
    num_samples = length(a.Data)./nChansInRawFile;
    RawData = a.Data;
    RawData = reshape(RawData,[128,length(a.Data)./nChansInRawFile]);
   
    if strcmp(animal, 'rEO_05')
        meanData = mean(RawData(dead_chs,:),1); % mean noise level
    elseif strcmp(animal, 'rEO_06')
        meanData = [];
    end
    
    %% Include impedance 
    if loadImpedance 
        if strcmp(animal, 'rEO_05')
            impedance_files = {...
                    'D:\Rat_Recording\rEO_05\session-1_230522_161409\Aright-Bleft-post-recovery-post-rec.csv', ...
                    'D:\Rat_Recording\rEO_05\session-2_230606_143306\Impedance-Aright-Bleft.csv', ...
                    'D:\Rat_Recording\rEO_05\session-3_230706_160753\session-3_impedances_Aright_Bleft_postrecording.csv', ...
                    'D:\Rat_Recording\rEO_05\session-4_230825_165102\session-4-Aright-Bleft.csv', ...
                    'D:\Rat_Recording\rEO_05\session-5_231005_161414\Aright-Bleft-postop-051023.csv'}; % manual input
            
            impedances_session = cell(numel(impedance_files),1);
            mean_ch_impedance = zeros(length(phys_ch_ord),1);
            for session_idx = 1:numel(impedance_files)
                session = impedance_files{session_idx};
                df = readtable(session, 'ReadVariableNames', true, 'PreserveVariableNames', true, 'NumHeaderLines', 0);
                df_A = df(1:64,:);
                impedances = df_A.("Impedance Magnitude at 1000 Hz (ohms)");
                impedances_session{session_idx} = impedances(phys_ch_ord);
            end
            mean_ch_impedance = mean(cell2mat(impedances_session'),2);  
            mean_ch_impedance(ismember(phys_ch_ord, dead_chs)) = [];
    
        elseif strcmp(animal, 'rEO_06')
            impedance_files = {fullfile(basepath, '6_impedances_eis', '1khz_a-right_b-left.csv')};
    
            impedances_session = cell(numel(impedance_files),1);
            mean_ch_impedance = zeros(length(phys_ch_ord),1);
            for session_idx = 1:numel(impedance_files)
                session = impedance_files{session_idx};
                df = readtable(session, 'ReadVariableNames', true, 'PreserveVariableNames', true, 'NumHeaderLines', 0);
                df_A = df(1:64,:);
                impedances = df_A.("Impedance Magnitude at 1000 Hz (ohms)");
                impedances_session{session_idx} = impedances(phys_ch_ord);
            end
            mean_ch_impedance = mean(cell2mat(impedances_session'),2); 
            if ~interpDeadChs
                mean_ch_impedance(ismember(phys_ch_ord, dead_chs)) = [];
            end
        end
    else
        mean_ch_impedance = [];
    end
    channelCC.impedance = mean_ch_impedance;
    
    %% Save parameters
    selected_channels = phys_ch_ord;
    selected_channels(ismember(selected_channels, dead_chs)) = [];
    
    channelCC.params.selected_chs = selected_channels;
    channelCC.params.basepath = basepath;
    channelCC.params.sessname = sessname;
    channelCC.params.filename = filename;
    channelCC.params.phys_ch_order = phys_ch_ord;
    channelCC.params.dead_chs = dead_chs;
    channelCC.params.loadImpedance = loadImpedance;
    channelCC.params.interpDeadChs = interpDeadChs;
    
    %% Select data
    % Load SWRs
    if exist(fullfile(basepath, sessname, [sessname '.curated_ripples.mat']))
        load(fullfile(basepath, sessname, [sessname '.curated_ripples.mat']));
    
        right_ripples_selected = ripple_timestamps(strcmp(ripple_classes(:),'right'),:);
        left_ripples_selected = ripple_timestamps(strcmp(ripple_classes(:),'left'),:);
        ripple_durations = ripple_timestamps(:,2) - ripple_timestamps(:,1);
    else
        error('This analysis needs the SWRs of the session.')
    end
    
    % Load theta windows or compute them here
    if exist(fullfile(basepath, sessname, [sessname '.theta_info.mat']))
        load(fullfile(basepath, sessname, [sessname '.theta_info.mat']));
        theta_windows = theta_info.theta_segments;
    else
        % Get raw LFP traces from selected channels
        LFPFromSelectedChannels = zeros(numel(theta_selected_channels), num_samples, 'int16');
    
        for i = 1:numel(theta_selected_channels)
            channelIdx = theta_selected_channels(i);
            disp(channelIdx);
            LFPFromSelectedChannels(i,:) = RawData(channelIdx:nChansInRawFile:end);
            LFPFromSelectedChannels = double(LFPFromSelectedChannels).*bitScaling;
        end
    
        % Find theta windows
        theta_windows = getThetaStates(LFPFromSelectedChannels,'freqlist',[2,20],'window',2,'noverlap',[2-1], ...
            'num_samples',num_samples,'sample_rate',fs,'f_theta',f_theta,'f_delta',[2 3], 'th2d_ratio_threshold',3);
        
        % Get the phase of the theta for each data point
        [thetaPhase, thetaLFP] = getThetaPhase(LFPFromSelectedChannels, 'sample_rate',fs, 'f_theta',f_theta);
    
        % Fine-tune the theta windows used 
        new_theta_windows = postprocessThetaSegments(theta_windows, LFPFromSelectedChannels, thetaLFP, thetaPhase, 'phase_threshold',15, ...
        'duration_threshold',0.5, 'sample_rate',fs, 'num_samples',num_samples);
    
        theta_windows = new_theta_windows;
    
        % Save theta info 
        theta_info.freqs = f_theta;
        theta_info.theta_segments = new_theta_windows;
        theta_info.thetaLFP = thetaLFP;
        theta_info.thetaPhase = thetaPhase;
        save(fullfile(basepath, sessname, [sessname '.theta_info.mat']),'theta_info');
    
    end
    
    theta_timestamps = theta_windows./fs;
    
    % Optional: to confirm the quality of the theta windows
    % test_theta(num_samples, fs, LFPFromSelectedChannels, new_theta_windows, thetaLFP, thetaPhase);
    
    %% Generate random OFF (non-SWR and non-theta) times
    ripple_durations = right_ripples_selected(:,2) - right_ripples_selected(:,1);
    theta_swr_windows = vertcat(theta_timestamps, right_ripples_selected);
    num_ripples = length(ripple_durations);
    
    % Select random segments of equal size as the SWRs
    num_samples = size(RawData,2);
    timestamps = linspace(0, num_samples/fs, num_samples);
    
    % Find segments without theta or SWRs
    try 
        event_range = timestamps(any(timestamps > theta_swr_windows(:,1) & timestamps < theta_swr_windows(:,2))); 
        valid_range = timestamps(~any(timestamps > theta_swr_windows(:,1) & timestamps < theta_swr_windows(:,2))); 
    catch
        disp('The arrays are too large. Using parallel processes instead.')
        if isempty(gcp("nocreate"))
            parpool;
        end
        valid_timestamps = true(1, num_samples);
        event_timestamps = false(1, num_samples);
        parfor i = 1:size(theta_swr_windows, 1)
            valid_timestamps = valid_timestamps & ~(timestamps > theta_swr_windows(i, 1) & timestamps < theta_swr_windows(i, 2));
            event_timestamps = event_timestamps | (timestamps > theta_swr_windows(i, 1) & timestamps < theta_swr_windows(i, 2)); 
        end
        valid_range = timestamps(valid_timestamps); % non-theta/non-SWR window (in sec)
        event_range = timestamps(event_timestamps); % theta and SWR window (in sec)
    end
    
    idx_perm = randperm(num_ripples);
    shuffled_ripple_durations = ripple_durations(idx_perm);
    
    random_start_time = valid_range(randperm(length(valid_range), num_ripples))';  % in sec
    random_end_time = random_start_time + shuffled_ripple_durations;
    
    % Resample if any new event falls in the invalid range
    event_mask = ismember(random_end_time, event_range) | ismember(random_start_time, event_range) | (random_end_time*fs) > num_samples;
    while any(event_mask)
        random_start_time = valid_range(randperm(length(valid_range), num_ripples));
        random_end_time = random_start_time + shuffled_ripple_durations';
        
        event_mask = ismember(random_end_times, event_range) | ismember(random_start_times, event_range);
    end
    shuffled_ripple_timestamps = [random_start_time, random_end_time]; 
    
    %% Get channel cross-correlation
    disp('Computing channel cross-correlation...');
    all_idx = 1:64;
    selected_idx = find(ismember(phys_ch_ord, selected_channels));
    dead_idx = find(ismember(phys_ch_ord, dead_chs));
    
    modes = ["swr","non_swr","theta"]; % time windows used 
    for i = 1:length(modes)
        mode = modes(i);
        
        switch mode
            case 'swr'
                mean_normalized_corr{1} = getChannelCC(ripple_timestamps, RawData, meanData, selected_channels, f_ripple, fs);
                mean_normalized_corr{2} = getChannelCC(ripple_timestamps, RawData, meanData, selected_channels, f_theta, fs);

                % Interpolate channel CC for dead channels 
                if interpDeadChs
                    mean_normalized_corr{1} = interpolate_corr_dead_chs(mean_normalized_corr{1}, selected_idx, dead_idx, all_idx);
                    mean_normalized_corr{2} = interpolate_corr_dead_chs(mean_normalized_corr{2}, selected_idx, dead_idx, all_idx);
                end
               
            case 'theta'
                mean_normalized_corr{1} = getChannelCC(theta_timestamps, RawData, meanData, selected_channels, f_ripple, fs);
                mean_normalized_corr{2} = getChannelCC(theta_timestamps, RawData, meanData, selected_channels, f_theta, fs);

                if interpDeadChs
                    mean_normalized_corr{1} = interpolate_corr_dead_chs(mean_normalized_corr{1}, selected_idx, dead_idx, all_idx);
                    mean_normalized_corr{2} = interpolate_corr_dead_chs(mean_normalized_corr{2}, selected_idx, dead_idx, all_idx);
                end
                
            case 'non_swr'        
                % Cross-correlation in non-SWR and non-theta segments
                mean_normalized_corr{1} = getChannelCC(shuffled_ripple_timestamps, RawData, meanData, selected_channels, f_ripple, fs);
                mean_normalized_corr{2} = getChannelCC(shuffled_ripple_timestamps, RawData, meanData, selected_channels, f_theta, fs);

                if interpDeadChs
                    mean_normalized_corr{1} = interpolate_corr_dead_chs(mean_normalized_corr{1}, selected_idx, dead_idx, all_idx);
                    mean_normalized_corr{2} = interpolate_corr_dead_chs(mean_normalized_corr{2}, selected_idx, dead_idx, all_idx);
                end
        end
    
        channelCC.(mode).ripple.corr = mean_normalized_corr{1};
        channelCC.(mode).theta.corr = mean_normalized_corr{2};
    
        channelCC.(mode).ripple.fband = f_ripple;
        channelCC.(mode).theta.fband = f_theta;
    end
    
    %% Get channel power
    disp('Computing channel power...')
    modes = ["swr","non_swr","theta"];
    for i = 1:length(modes)
        mode = modes(i);
    
        switch mode
            case 'swr'
                mean_channel_power{1} = getChannelPower(ripple_timestamps, RawData, selected_channels, f_ripple, fs);
                mean_channel_power{2} = getChannelPower(ripple_timestamps, RawData, selected_channels, f_theta, fs);
        
                if interpDeadChs
                    mean_channel_power{1} = interpolate_power_dead_chs(mean_channel_power{1}, selected_idx, dead_idx, all_idx);
                    mean_channel_power{2} = interpolate_power_dead_chs(mean_channel_power{2}, selected_idx, dead_idx, all_idx);
                end

            case 'theta'
                mean_channel_power{1} = getChannelPower(theta_timestamps, RawData, selected_channels, f_ripple, fs);
                mean_channel_power{2} = getChannelPower(theta_timestamps, RawData, selected_channels, f_theta, fs);

                if interpDeadChs
                    mean_channel_power{1} = interpolate_power_dead_chs(mean_channel_power{1}, selected_idx, dead_idx, all_idx);
                    mean_channel_power{2} = interpolate_power_dead_chs(mean_channel_power{2}, selected_idx, dead_idx, all_idx);
                end
                    
            case 'non_swr'
                mean_channel_power{1} = getChannelPower(shuffled_ripple_timestamps, RawData, selected_channels, f_ripple, fs);
                mean_channel_power{2} = getChannelPower(shuffled_ripple_timestamps, RawData, selected_channels, f_theta, fs);

                if interpDeadChs
                    mean_channel_power{1} = interpolate_power_dead_chs(mean_channel_power{1}, selected_idx, dead_idx, all_idx);
                    mean_channel_power{2} = interpolate_power_dead_chs(mean_channel_power{2}, selected_idx, dead_idx, all_idx);
                end
        end
        
        channelCC.(mode).ripple.power = mean_channel_power{1};
        channelCC.(mode).theta.power = mean_channel_power{2};
    end

    %% Plot the channel cross-correlation and power
    plotChannelCC('basepath',basepath,'sessname',sessname,'channelCC',channelCC, ...
        'signals',{"ripple","theta"},'modes',{"swr","theta"}, ...
        'fbands',{strjoin(string(f_ripple), '-'),strjoin(string(f_theta), '-')},'norm','zscore');
    
    %% Save cross-correlation 
    if interpDeadChs
        save(fullfile(basepath, sessname, strcat(sessname, '.channelCC_allChs', '.mat')), 'channelCC');
    else
        save(fullfile(basepath, sessname, strcat(sessname, '.channelCC', '.mat')), 'channelCC');
    end
end


%% Helper functions 
function mean_normalized_corr = interpolate_corr_dead_chs(corr, selected_idx, dead_idx, all_idx) 
    C_full = NaN(64, 64);
    C_full(selected_idx, selected_idx) = corr;
    
    % Interpolate missing values for the dead channels
    [x, y] = meshgrid(all_idx, all_idx);
    % [sel_x, sel_y] = meshgrid(selected_idx, selected_idx);
    
    known_points_x = x(~isnan(C_full));
    known_points_y = y(~isnan(C_full));
    known_values = C_full(~isnan(C_full));
    
    C_full = griddata(known_points_x, known_points_y, known_values, x, y, 'cubic');
    mean_normalized_corr = C_full;
end

function mean_channel_power = interpolate_power_dead_chs(power, selected_idx, dead_idx, all_idx) 
    V_full = NaN(64, 1);
    V_full(selected_idx) = power;
     
    % Perform interpolation
    V_full = interp1(selected_idx, power, all_idx, 'spline');
    mean_channel_power = V_full;
end