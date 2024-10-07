function firingRate = getFiringRate(varargin)

%% Set parameters
p=inputParser;
addParameter(p,'basepath',pwd,@isfolder);
addParameter(p,'sample_rate',20000,@isnumeric);
addParameter(p,'nChansInRawFile',128,@isnumeric);
addParameter(p,'clusterSites',[],@isnumeric);

parse(p,varargin{:});
basepath = p.Results.basepath;
sample_rate = p.Results.sample_rate;
nChansInRawFile = p.Results.nChansInRawFile;
clusterSites = p.Results.clusterSites;

%% Load data
% Curated SWRs
if ~isempty(fullfile(basepath, '*.curated_ripples.mat'))
    file = dir(fullfile(basepath, '*.curated_ripples.mat'));
    load(file(1).name);
    num_ripples = length(ripple_classes);
else
    error('The curated ripple file (CNN detection) does not exist in this directory.');
end

% Theta info 
if ~isempty(fullfile(basepath, '*.theta_info_new.mat')) 
    disp('Theta info detected. Loading...');
    file = dir(fullfile(basepath, '*.theta_info_new.mat'));
    load(file(1).name);
else
    error('Theta info file does not exist in this directory.');
end

% Theta firing
if ~isempty(fullfile(basepath, '*.theta_firing_new.cellinfo.mat')) 
    disp('Theta firing already detected. Loading...');
    file = dir(fullfile(basepath, '*.theta_firing_new.cellinfo.mat'));
    load(file(1).name);
else
    error('Theta firing file does not exist in this directory.');
end

% SWR firing
if ~isempty(fullfile(basepath, '*.swr_firing_new.cellinfo.mat')) 
    disp('SWR firing already detected. Loading...');
    file = dir(fullfile(basepath, '*.swr_firing_new.cellinfo.mat'));
    load(file(1).name);
else
    error('SWR firing file does not exist in this directory.');
end

% Spikes
spikes = loadSpikes_JRC('basepath',basepath,'forceReload',false);
num_neurons = length(spikes.UID);

% Raw data 
if contains(basepath, 'concatenated')
    raw_file = fullfile(basepath, '128ch_concat_data.dat');
else
    raw_file = fullfile(basepath, 'amplifier.dat'); 
end
FileInfo = dir(raw_file);
num_samples = FileInfo.bytes/(nChansInRawFile * 2); % int16 = 2 bytes

% Spiking info
if isempty(clusterSites)
    if contains(basepath, 'concatenated')
        load(fullfile(basepath, 'concat_data_res.mat'), 'clusterSites');
    else
        load(fullfile(basepath, 'amplifier_res.mat'), 'clusterSites');
    end
end

%% Calculate firing rates
firing_rate = zeros(num_neurons,1);
firing_rate_swr = zeros(num_neurons,1);
firing_rate_theta = zeros(num_neurons,1);

% Duration of all SWRs together for each hemisphere
ripples_right = find(strcmp(ripple_classes, 'right'));
ripples_left = find(strcmp(ripple_classes, 'left'));

swr_duration_right = sum((ripple_timestamps(ripples_right,2) - ripple_timestamps(ripples_right,1)));
swr_duration_left = sum((ripple_timestamps(ripples_left,2) - ripple_timestamps(ripples_left,1)));

% Duration of theta segment
theta_duration = sum((theta_info.theta_segments(:,2) - theta_info.theta_segments(:,1))/sample_rate);

for c = 1:num_neurons
    firing_rate(c) = length(spikes.times{spikes.cluster_index == c}) / (num_samples/sample_rate);  % overall firing rate

    num_swr_spikes = sum(swr_firing{c}.spikes_start(:,:), [1,2]) + sum(swr_firing{c}.spikes_end(:,:), [1,2]);
    if clusterSites(c) < 65
        firing_rate_swr(c) = num_swr_spikes / swr_duration_right;
    else
        firing_rate_swr(c) = num_swr_spikes / swr_duration_left;
    end

    num_theta_spikes = sum(theta_firing.spikes(c,:), 2);
    firing_rate_theta(c) = num_theta_spikes / theta_duration;
end

firingRate.firing_rate = firing_rate;
firingRate.firing_rate_swr = firing_rate_swr;
firingRate.firing_rate_theta = firing_rate_theta;

end