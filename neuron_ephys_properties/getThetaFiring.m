function [theta_firing] = getThetaFiring(varargin)

% The firing probability is the sum of spikes for each phase bin divided by the sum of spikes during the chosen theta segment. 

%% Set parameters
p=inputParser;
addParameter(p,'basepath',pwd,@isstr);
addParameter(p,'forceReload',true,@islogical);
addParameter(p,'num_theta_bins',18,@isnumeric);
addParameter(p,'bin_edges_theta',[],@isnumeric);
addParameter(p,'num_neurons',[],@isnumeric);

parse(p,varargin{:});
basepath = p.Results.basepath;
forceReload = p.Results.forceReload;
num_theta_bins = p.Results.num_theta_bins;
bin_edges_theta = p.Results.bin_edges_theta;
num_neurons = p.Results.num_neurons;

%% Check if theta firing has already been calculated 
if ~isempty(fullfile(basepath, '*.theta_firing_new.cellinfo.mat')) & ~forceReload
    disp('Theta firing already determined. Loading...');
    file = dir(fullfile(basepath, '*.theta_firing_new.cellinfo.mat'));
    load(file(1).name);
    return 
end

%% Load data 
% Theta info
if ~isempty(fullfile(basepath, '*.theta_info_new.mat'))  
    disp('Theta info detected. Loading...');
    file = dir(fullfile(basepath, '*.theta_info_new.mat'));
    load(file(1).name);
    theta_windows = theta_info.theta_segments;
    thetaPhase = theta_info.thetaPhase;
else
    error('No theta data were found in this directory.');
end

% Spike and electrode site data
if contains(basepath,'concatenated')
    load(fullfile(basepath, 'concat_data_res.mat'), 'spikeTimes');
    load(fullfile(basepath, 'concat_data_res.mat'), 'spikeClusters');
    load(fullfile(basepath, 'concat_data_res.mat'), 'clusterSites');
else
    load(fullfile(basepath, 'amplifier_res.mat'), 'spikeTimes');
    load(fullfile(basepath, 'amplifier_res.mat'), 'spikeClusters');
    load(fullfile(basepath, 'amplifier_res.mat'), 'clusterSites');
end

%% Start analysis 
p_theta = zeros(num_neurons, 1);
theta_spikes = zeros(num_neurons, num_theta_bins);
theta_firing_prob = zeros(num_neurons, num_theta_bins);

num_theta_windows = length(theta_windows);
if isempty(num_theta_windows)
    error('No theta windows have been found in this session.');
end

for c = 1:num_neurons
    % Convert spike times from seconds to sample indices
    spikeIndices = double(spikeTimes(spikeClusters == c));

    % Extract spikes by the start-stop of the theta
    % spikeIndices_selected = arrayfun(@(idx) spikeIndices(find(start_theta_sample{idx} < spikeIndices & spikeIndices < stop_theta_sample{idx})), ...
    %     1:length(theta_windows), 'UniformOutput', false);
    spikeIndices_selected = arrayfun(@(idx) spikeIndices(theta_windows(idx,1) < spikeIndices & spikeIndices < theta_windows(idx,end)), ...
        1:num_theta_windows, 'UniformOutput', false);

    % Extract theta phase at each spike time
    theta_spike_phases = [];
    win_spikes = zeros(num_theta_windows, num_theta_bins);
    win_firing_prob = zeros(num_theta_windows, num_theta_bins);

    for w = 1:num_theta_windows
        if ~isempty(spikeIndices_selected{w})
            if clusterSites(c) < 65
                spikePhases = thetaPhase(spikeIndices_selected{w},2); % right shank
            else
                spikePhases = thetaPhase(spikeIndices_selected{w},1); % left shank
            end
            win_spikes(w,:) = histcounts(spikePhases, bin_edges_theta);
            win_firing_prob(w,:) = win_spikes(w,:) ./ length(spikeIndices_selected{w});
            theta_spike_phases = [theta_spike_phases spikePhases'];
        end
    end
    theta_phases{c} = theta_spike_phases;
    theta_spikes(c,:) = mean(win_spikes,1);
    theta_firing_prob(c,:) = mean(win_firing_prob,1);
end

%% Circular statistics (Rayleigh test)
for c = 1:num_neurons
    if ~isempty(theta_phases{c})
        theta_phases_rad{c} = circ_ang2rad(theta_phases{c});
        p_theta(c) = circ_rtest(theta_phases_rad{c});
    else
        p_theta(c) = NaN;
    end
end

%% Save results
theta_firing.phases = theta_phases; % phase of each spike
theta_firing.phases_rad = theta_phases_rad;
theta_firing.spikes = theta_spikes; % mean number of spikes per bin
theta_firing.firing_prob = theta_firing_prob; % mean firing prob
theta_firing.p_values = p_theta;
theta_firing.num_bins = num_theta_bins;

[~,sessname] = fileparts(basepath);
save(fullfile(basepath, [sessname '.theta_firing_new.cellinfo.mat']),'theta_firing');