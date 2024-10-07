% This script calculates the cross-correlation among the recorded channels
% for specific frequency bands and time windows during the recording.
% Commonly, the recording is split between SWR times and non-SWR times and
% the power of the signal is used for the cross-correlation calculation. 
%
% OUTPUT
% channelCC struct
%   - sig_channels          - chs w/ significantly higher power during SWRs
%   - (mode).corr           - channel CC in mode SWR/non-SWR
%   - (mode).fband          - frequency band to compute power and CC
%   - (mode).power          - channel power in mode SWR/non-SWR
%   - MRImap_borders        - map of structures according to MRI map
%   - impedance             - impedance of selected channels
%   - params                - information on parameters of the script
% 
% HISTORY 
% Athina Apostolelli 2024
% Adapted from Eminhan Ozil

%% Set parameters
basepath = 'D:\Rat_Recording\rEO_05';
sessname = '128ch_concatenated_sessions';
filename = fullfile(basepath, sessname, '128ch_concat_data.dat');
animal = 'rEO_05';

MRImap = false;
loadImpedance = true;
plotImpedance = false;
power = true;
fs=20000;
sig_channelsL = [];
sig_channelsH = [];
f_ripple = [150,250];
f_theta = [4,10];
theta_selected_channels = [36]+1;

% Raw data is not ordered
nChansInRawFile = 128;
bitScaling = 0.195;
a=memmapfile(filename, 'Format','int16');
num_samples = length(a.Data)./nChansInRawFile;
RawData=a.Data;
RawData=reshape(RawData,[128,length(a.Data)./nChansInRawFile]);

% selected_noise=[22 24 28 50 4 63 62]+1

%%
phys_ch_ord = [20 21 41 18 40 19 43 16 42 23 45 22 44 25 38 24 39 27 36 26 37 29 34 28 35 31 32 17 33 15 30 14 46 13 47 12 49 11 48 10 51 9 50 8 53 7 52 6 55 5 54 4 57 3 56 2 59 1 58 0 61 63 60 62]+1;
dead_chs = [20,21,22,25,24,28,31,30,14,13,4,63,62,50,8]+1; % concat data rEO_05
% dead_chs = [18,16,22,23,44,39,36,32,13,49,54,8,53,7,52,6]+1;  % right rEO_06
% dead_chs = [99,96,97,76,74,72]+1;  % left rEO_06
meanData = mean(RawData(dead_chs,:),1); 

selected_channels = phys_ch_ord;
selected_channels(ismember(selected_channels, dead_chs))=[];

channelCC.params.selected_chs = selected_channels;
channelCC.params.basepath = basepath;
channelCC.params.sessname = sessname;
channelCC.params.filename = filename;
channelCC.params.phys_ch_order = phys_ch_ord;
channelCC.params.dead_chs = dead_chs;
channelCC.params.plotMRImap = MRImap;
channelCC.params.plotImpedance = plotImpedance;
channelCC.params.loadImpedance = loadImpedance;
channelCC.params.plotPower = power;

channelCC.sig_channels.low = sig_channelsL; 
channelCC.sig_channels.high = sig_channelsH; 

%% Select data
% Load SWRs
if exist(fullfile(basepath, sessname, [sessname '.curated_ripples.mat']))
    load(fullfile(basepath, sessname, [sessname '.curated_ripples.mat']));

    right_ripples_selected = ripple_timestamps(strcmp(ripple_classes(:),'right'),:);
    left_ripples_selected = ripple_timestamps(strcmp(ripple_classes(:),'left'),:);
    ripple_durations = ripple_timestamps(:,2) - ripple_timestamps(:,1);
    num_ripples = length(ripple_timestamps);
else
    right_ripples_selected = [];
    left_ripples_selected = [];
    ripple_durations = [];
    num_ripples = [];
end

% Load theta
if exist(fullfile(basepath, sessname, [sessname '.theta_info.mat']))
    load(fullfile(basepath, sessname, [sessname '.theta_info.mat']));
    theta_windows = theta_info.theta_segments;
else
    LFPFromSelectedChannels = zeros(numel(theta_selected_channels), num_samples, 'int16');

    for i = 1:numel(theta_selected_channels)
        channelIdx = theta_selected_channels(i);
        disp(channelIdx);
        LFPFromSelectedChannels(i,:) = RawData(channelIdx:nChansInRawFile:end);
        LFPFromSelectedChannels = double(LFPFromSelectedChannels).*bitScaling;
    end

    % Extract theta
    theta_windows = getThetaStates(LFPFromSelectedChannels,'freqlist',[2,20],'window',2,'noverlap',[2-1], ...
        'num_samples',num_samples,'sample_rate',fs,'f_theta',[4 10],'f_delta',[2 3], 'th2d_ratio_threshold',3);
    
    [thetaPhase, thetaLFP] = getThetaPhase(LFPFromSelectedChannels, 'sample_rate',fs, 'f_theta',f_theta);

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

% Remove short theta windows
theta_timestamps = theta_windows./fs;

% Random time window
% This time window contains 2 SWRs 
% time_window = [2932, 2936.900]; % SWRs segment
time_window = [3207, 3211.700]; % theta segment
% time_window = [209.780, 210.380];


%% Random OFF (SWR and theta) times
ripple_durations = right_ripples_selected(:,2) - right_ripples_selected(:,1);
theta_swr_windows = vertcat(theta_timestamps, right_ripples_selected);

% Select random segments of equal size as the SWRs
num_samples = size(RawData,2);
timestamps = linspace(0, num_samples/fs, num_samples);
event_range = timestamps(any(timestamps > theta_swr_windows(:,1) & timestamps < theta_swr_windows(:,2))); % theta and SWR window (in sec)
valid_range = timestamps(~any(timestamps > theta_swr_windows(:,1) & timestamps < theta_swr_windows(:,2))); % non-theta/non-SWR window (in sec)

idx_perm = randperm(num_ripples);
shuffled_ripple_durations = ripple_durations(idx_perm);

random_start_time = valid_range(randperm(length(valid_range), num_ripples))';  % in sec
random_end_time = random_start_time + shuffled_ripple_durations;

event_mask = ismember(random_end_time, event_range) | ismember(random_start_time, event_range) | (random_end_time*fs) > num_samples;
while any(event_mask)
    random_start_time = valid_range(randperm(length(valid_range), num_ripples));
    random_end_time = random_start_time + shuffled_ripple_durations';
    % Resample if any new event falls in the invalid range
    event_mask = ismember(random_end_times, event_range) | ismember(random_start_times, event_range);
end
shuffled_ripple_timestamps = [random_start_time, random_end_time]; 

%% Cross-correlation
modes = ["swr","non_swr","theta"];
for i = 1:length(modes)
    mode = modes(i);
    % [mean_normalized_corr] = getChannelCC(selected_channels, ripple_timestamps, shuffled_ripple_timestamps, RawData, mode, fs, theta_timestamps, time_window, f_ripple, f_theta);
    switch mode
        case 'swr'
            mean_normalized_corr{1} = getChannelCC(ripple_timestamps, RawData, meanData, selected_channels, f_ripple, fs);
            mean_normalized_corr{2} = getChannelCC(ripple_timestamps, RawData, meanData, selected_channels, f_theta, fs);
           
        case 'theta'
            mean_normalized_corr{1} = getChannelCC(theta_timestamps, RawData, meanData, selected_channels, f_ripple, fs);
            mean_normalized_corr{2} = getChannelCC(theta_timestamps, RawData, meanData, selected_channels, f_theta, fs);
            
        case 'non_swr'        
            % Cross-correlation in non-SWR and non-theta segments
            mean_normalized_corr{1} = getChannelCC(shuffled_ripple_timestamps, RawData, meanData, selected_channels, f_ripple, fs);
            mean_normalized_corr{2} = getChannelCC(shuffled_ripple_timestamps, RawData, meanData, selected_channels, f_theta, fs);
    
    end
    channelCC.(mode).ripple.corr = mean_normalized_corr{1};
    channelCC.(mode).ripple.fband = f_ripple;
    channelCC.(mode).theta.corr = mean_normalized_corr{2};
    channelCC.(mode).theta.fband = f_theta;
end

%% Include the MRI channel map to compare imaging with ephys
if MRImap
    mri_map_borders = [1,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]; % manual input (1 = brain structure change)
    for i = 1:length(mri_map_borders)
        if ismember(phys_ch_ord(i), dead_chs) & mri_map_borders(i)==1
            if ~isnan(mri_map_borders(i+1))
                mri_map_borders(i+1) = 1;
            end
            mri_map_borders(i) = nan;
        end
    end
    mri_map_borders(ismember(phys_ch_ord, dead_chs)) = [];
    mri_map_borders(isnan(mri_map_borders)) = [];
    
else
    mri_map_borders = [];
end
channelCC.MRImap_borders = mri_map_borders;

%% Include impedance 
if loadImpedance 
    if strcmp(animal, 'rEO_05')
        impedance_files = {...
                'D:\Rat_Recording\rEO_05\session-1_230522_161409\Aright-Bleft-post-recovery-post-rec.csv', ...
                'D:\Rat_Recording\rEO_05\session-2_230606_143306\Impedance-Aright-Bleft.csv', ...
                'D:\Rat_Recording\rEO_05\session-3_230706_160753\session-3_impedances_Aright_Bleft_postrecording.csv', ...
                'D:\Rat_Recording\rEO_05\session-4_230825_165102\session-4-Aright-Bleft.csv', ...
                'D:\Rat_Recording\rEO_05\session-5_231005_161414\Aright-Bleft-postop-051023.csv'};
        
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
        mean_ch_impedance(ismember(phys_ch_ord, dead_chs)) = [];
    end
else
    mean_ch_impedance = [];
end
channelCC.impedance = mean_ch_impedance;

%% Mean channel power
modes = ["swr","non_swr","theta"];
for i = 1:length(modes)
    mode = modes(i);
    mean_channel_power = getChannelPower(selected_channels, right_ripples_selected, shuffled_ripple_timestamps, theta_timestamps, mode, RawData, fs, f_ripple, f_theta);
    channelCC.(mode).ripple.power = mean_channel_power{1};
    channelCC.(mode).theta.power = mean_channel_power{2};
end

%% Plot the cross-correlation
for i = 1:length(modes)
    mode = modes(i);
    plotChannelCC(channelCC, mode);
end
%% Statistics
% To determine which channels change significantly their power during the
% SWRs, the power of each channel during SWRs is compared with that of a
% distribution from 100 (called shuffles below) power calculations during 
% non-SWR times. 

% timestamps = linspace(0, num_samples/fs, num_samples);
% event_range = timestamps(any(timestamps > ripple_timestamps(:,1) & timestamps < ripple_timestamps(:,2))); % SWR window (in sec)
% valid_range = timestamps(~any(timestamps > ripple_timestamps(:,1) & timestamps < ripple_timestamps(:,2))); % non-SWR window (in sec)

%% Cross-correlation in shuffled time windows
% num_shuffles = 5;
% 
% shuffled_ripple_durations = zeros(num_shuffles, num_ripples);
% shuffled_ripple_timestamps = zeros(num_shuffles, num_ripples, 2);
% shuffled_ripple_classes = cell(num_shuffles, num_ripples);
% random_start_time = zeros(num_shuffles, num_ripples);
% random_end_time = zeros(num_shuffles, num_ripples);
% shuffled_right_ripples_selected = zeros(num_shuffles, length(right_ripples_selected), 2);
% 
% for s = 1:num_shuffles
%     idx_perm = randperm(num_ripples);
%     shuffled_ripple_durations(s,:) = ripple_durations(idx_perm);
%     shuffled_ripple_classes(s,:) = ripple_classes(idx_perm);
% 
%     random_start_time(s,:) = valid_range(randperm(length(valid_range), num_ripples));  % in sec
%     random_end_time(s,:) = random_start_time(s,:) + shuffled_ripple_durations(s,:);
% 
%     event_mask = ismember(random_end_time(s,:), event_range) | ismember(random_start_time(s,:), event_range);
%     while any(event_mask)
%         random_start_time(s,:) = valid_range(randperm(length(valid_range), num_ripples));
%         random_end_time(s,:) = random_start_time(s,:) + shuffled_ripple_durations(s,:)';
%         % Resample if any new event falls in the invalid range
%         event_mask = ismember(random_end_times(s,:), event_range) | ismember(random_start_times(s,:), event_range);
%     end
%     shuffled_ripple_timestamps(s,:,:) = [random_start_time(s,:)', random_end_time(s,:)'];   
%     shuffled_right_ripples_selected(s,:,:) = shuffled_ripple_timestamps(s,strcmp(shuffled_ripple_classes(s,:),'right'),:);
% end
% 
% % Mean across ripples
% shuffled_mean_normalized_corr = arrayfun(@(idx) getChannelCC(selected_channels, squeeze(shuffled_right_ripples_selected(idx,:,:)), RawData, mode, fs, theta_info, time_window), 1:num_shuffles, 'UniformOutput',false);    
% 
% % Mean across shuffles
% mean_shuffled_normalzied_corr = mean(cat(3,shuffled_mean_normalized_corr{:}), 3); 
% std_shuffled_normalized_corr = std(cat(3,shuffled_mean_normalized_corr{:}),[],3);
% 
% % Plot the cross-correlogram
% figure;
% imagesc(mean_shuffled_normalzied_corr)
% colorbar;

%% Significant channels


%% Matrix correlation across conditions

%% Shuffle during non-SWR windows 
% % Takes a few hours to complete 
% mode = 'non_swr';
% num_shuffles = 100;
% 
% shuffled_ripple_durations = zeros(num_shuffles, num_ripples);
% shuffled_ripple_timestamps = zeros(num_shuffles, num_ripples, 2);
% shuffled_ripple_classes = cell(num_shuffles, num_ripples);
% random_start_time = zeros(num_shuffles, num_ripples);
% random_end_time = zeros(num_shuffles, num_ripples);
% shuffled_right_ripples_selected = zeros(num_shuffles, length(right_ripples_selected), 2);
% 
% for s = 1:num_shuffles
%     idx_perm = randperm(num_ripples);
%     shuffled_ripple_durations(s,:) = ripple_durations(idx_perm);
%     shuffled_ripple_classes(s,:) = ripple_classes(idx_perm);
% 
%     random_start_time(s,:) = valid_range(randperm(length(valid_range), num_ripples));  % in sec
%     random_end_time(s,:) = random_start_time(s,:) + shuffled_ripple_durations(s,:);
% 
%     event_mask = ismember(random_end_time(s,:), event_range) | ismember(random_start_time(s,:), event_range) | random_end_time(s,:)*fs > size(RawData,2);
%     while any(event_mask)
%         random_start_time(s,:) = valid_range(randperm(length(valid_range), num_ripples));
%         random_end_time(s,:) = random_start_time(s,:) + shuffled_ripple_durations(s,:);
%         % Resample if any new event falls in the invalid range
%         event_mask = ismember(random_end_times(s,:), event_range) | ismember(random_start_times(s,:), event_range);
%     end
%     shuffled_ripple_timestamps(s,:,:) = [random_start_time(s,:)', random_end_time(s,:)'];   
%     shuffled_right_ripples_selected(s,:,:) = shuffled_ripple_timestamps(s,strcmp(shuffled_ripple_classes(s,:),'right'),:);
% end
% 
% shuffled_mean_channel_power = arrayfun(@(idx) getChannelPower(selected_channels, squeeze(shuffled_right_ripples_selected(idx,:,:)), RawData, fs, fband), 1:num_shuffles, 'UniformOutput',false);
% 
% mean_shuffled_channel_power = mean(cell2mat(shuffled_mean_channel_power),2); % mean across shuffles
% % std_shuffled_channel_power = std(cell2mat(shuffled_mean_channel_power),[],2);
% 
% channelCC.(mode).power = mean_shuffled_channel_power;
% 
% %% Find channels that change significantly during SWRs
% percentiles = prctile(channelCC.non_swr.power, [2.5, 97.5]);
% sig_channelsL = channelCC.params.selected_chs(find(channelCC.swr.power <= percentiles(1)));
% sig_channelsH = channelCC.params.selected_chs(find(channelCC.swr.power >= percentiles(2)));
% channelCC.sig_channels.low = sig_channelsL;
% channelCC.sig_channels.high = sig_channelsH;
% 
% %% Plot the cross-correlation with significant channels indicated 
% plotChannelCC(channelCC, 'swr');

%% Save cross-correlation 
save(fullfile(basepath, sessname, strcat(sessname, '.channelCC', '.mat')), 'channelCC');










