function [swr_firing, shuffled_swr_firing, neuron_class_bySWR] = getSWRFiringWrapper(varargin)

%% Set parameters
p = inputParser;
addParameter(p,'basepath',pwd,@ischar);
addParameter(p,'forceReload',true,@islogical);
addParameter(p,'pathLFP','',@ischar);
addParameter(p,'nChansInLFPFile',16,@isnumeric);
addParameter(p,'num_shuffles',100,@isnumeric);
addParameter(p,'sample_rate',20000,@isnumeric);
addParameter(p,'sample_rate_LFP',2000,@isnumeric);
addParameter(p,'pyramidal_layer',[],@isnumeric);
addParameter(p,'mode',{'swr','shuffle'},@iscell);
addParameter(p,'classifyNeurons',false,@islogical);

parse(p,varargin{:});
basepath = p.Results.basepath;
forceReload = p.Results.forceReload;
pathLFP = p.Results.pathLFP;
nChansInLFPFile = p.Results.nChansInLFPFile;
num_shuffles = p.Results.num_shuffles;
sample_rate = p.Results.sample_rate;
sample_rate_LFP = p.Results.sample_rate_LFP;
pyramidal_layer = p.Results.pyramidal_layer;
mode = p.Results.mode;
classifyNeurons = p.Results.classifyNeurons;

[~,basename] = fileparts(basepath);

%% Check if SWR properties have already been calculated
if ~isempty(dir(fullfile(basepath, '*.swr_firing_new.cellinfo.mat'))) & ~forceReload
    disp('SWR firing already detected. Loading...');
    file = dir(fullfile(basepath, '*.swr_firing_new.cellinfo.mat'));
    load(file(1).name);
    return 
end

%% Load data
% Theta info 
if ~isempty(fullfile(basepath, '*.theta_info_new.mat')) 
    disp('Theta info detected. Loading...');
    file = dir(fullfile(basepath, '*.theta_info_new.mat'));
    load(file(1).name);
end

% Curated SWRs
if ~isempty(fullfile(basepath, '*.curated_ripples.mat'))
    file = dir(fullfile(basepath, '*.curated_ripples.mat'));
    load(file(1).name);
    num_ripples = length(ripple_classes);
else
    error('The curated ripple file (CNN detection) does not exist in this directory.');
end

% Spikes and electrode site data 
spikes = loadSpikes_JRC('basepath',basepath,'forceReload',false);
num_neurons = length(spikes.UID);

if contains(basepath,'concatenated')
    load(fullfile(basepath, 'concat_data_res.mat'), 'clusterSites');
else
    load(fullfile(basepath, 'amplifier_res.mat'), 'clusterSites');
end

% Load and filter LFP (downsampled and filtered)
FileInfo = dir(pathLFP);
LFP = ImporterDAT_multi(fullfile(FileInfo.folder, FileInfo.name), nChansInLFPFile, [1:nChansInLFPFile]);
num_samples = FileInfo.bytes/(nChansInLFPFile * 2); % per channel

d = designfilt('bandpassfir','FilterOrder',600,'StopbandFrequency1',125,'PassbandFrequency1',130,'PassbandFrequency2',245,'StopbandFrequency2',250,'SampleRate',2000);
LFP_detrended = detrend(double(LFP')); % raw
LFP_filtered = filtfilt(d, LFP_detrended); % ripples
LFP_filtered_hilbert = abs(hilbert(LFP_filtered)); % hilbert transform

%% Get firing for SWRs
swr_firing = getSWRfiring(ripple_timestamps, ripple_classes, spikes, LFP_filtered_hilbert, clusterSites, 'num_samples',num_samples,'sample_rate',sample_rate, ...
    'sample_rate_LFP',sample_rate_LFP,'pyramidal_layer',pyramidal_layer,'mode',mode{1});

%% Get firing for shuffled SWRs 
% Concatenate theta windows and SWR events
theta_swr_windows = vertcat(theta_info.theta_segments, ripple_timestamps);

ripple_durations = ripple_timestamps(:,2) - ripple_timestamps(:,1);
timestamps = linspace(0, num_samples/sample_rate, num_samples);

% Define non-theta/non-SWR epochs where random time windows will be chosen
event_range = timestamps(any(timestamps > theta_swr_windows(:,1) & timestamps < theta_swr_windows(:,2))); % theta and SWR window (in sec)
valid_range = timestamps(~any(timestamps > theta_swr_windows(:,1) & timestamps < theta_swr_windows(:,2))); % non-theta/non-SWR window (in sec)

% Perform the shuffles
shuffled_swr_firing = getShuffledSWRfiring(ripple_timestamps, ripple_classes, ripple_durations, spikes, LFP_filtered_hilbert, event_range, valid_range, ...
    num_shuffles, num_ripples, clusterSites, 'num_samples',num_samples,'sample_rate',sample_rate, 'sample_rate_LFP',sample_rate_LFP, ...
    'pyramidal_layer',pyramidal_layer,'mode',mode{2});

%% Classify neurons according to SWR firing
if classifyNeurons
    neuron_class_bySWR = classifyNeurons_SWRfiring(swr_firing, shuffled_swr_firing, num_neurons, num_shuffles, 'class1_win_start',2, ...
        'class1_win_dur',6, 'class2_win1_start',8, 'class2_win1_dur',4, 'class2_win2_start',14, 'class2_win2_dur',10);
else
    neuron_class_bySWR = [];
end

%% Calculate statistics of SWR histograms 
for c = 1:num_neurons
    shuffled_all_firing_prob = arrayfun(@(idx) [shuffled_swr_firing{c}{idx}.first_firing_prob, shuffled_swr_firing{c}{idx}.firing_prob_start, ...
        shuffled_swr_firing{c}{idx}.firing_prob_end, shuffled_swr_firing{c}{idx}.second_firing_prob], ...
        1:num_shuffles, 'UniformOutput', false);

    % Mean across ripples
    mean_shuffled_firing_prob = arrayfun(@(idx) mean(shuffled_all_firing_prob{idx},1), 1:num_shuffles, 'UniformOutput', false);
    mean_shuffled_firing_prob_array = reshape(cell2mat(mean_shuffled_firing_prob),[28, num_shuffles]); % 10 + 8 + 10 = 28 bins

    % Mean across shuffles
    mean_firing_prob_shuffled = mean(mean_shuffled_firing_prob_array,2)';
    std_firing_prob_shuffled = std(mean_shuffled_firing_prob_array,[],2)';

    swr_firing{c}.mean_firing_prob_shuffled = mean_firing_prob_shuffled;
    swr_firing{c}.std_firing_prob_shuffled = std_firing_prob_shuffled;
end

%% Save all results
save(fullfile(basepath, [basename '.swr_firing_new.cellinfo.mat']), 'swr_firing', 'shuffled_swr_firing', 'neuron_class_bySWR');

end