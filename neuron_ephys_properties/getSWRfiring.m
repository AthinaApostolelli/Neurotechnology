function [swr_firing] = getSWRfiring(ripple_timestamps, ripple_classes, spikes, LFP_filtered_hilbert, clusterSites, varargin)

% This function calculates the firing probability of a neuron during, and 
% around a SWR. 
% The bins within the SWRs have a variable length. In the time window from 
% SWR start [-1] to peak [0] (max amplitude of ripple - determined from the 
% Hilbert transform), all bins have the same size, which is different from 
% the size of the bins in the time window from SWR peak [0] to end [1], 
% which again have the same size. 
% The bins around the SWRs have the same length each. (Default) 10 bins 
% before and 10 bins after the events are considered. The firing 
% probability is the sum of spikes for each time bin divided by the sum of 
% spikes during the entire recording. 
%
% NOTE: the function is created for data from bilateral recordings
%
% Written by Athina Apostolelli - aapostolelli@ethz.ch

%% Set input parameters
p=inputParser;
addParameter(p,'basepath',pwd,@ischar);
addParameter(p,'num_samples',[],@isnumeric);
addParameter(p,'sample_rate',20000,@isnumeric);
addParameter(p,'sample_rate_LFP',2000,@isnumeric);
addParameter(p,'pyramidal_layer',[],@isnumeric);
addParameter(p,'num_bins_around',10,@isnumeric);
addParameter(p,'mode','swr',@ischar); % shuffle or swr
addParameter(p,'saveMat',true,@islogical); % TODO

parse(p,varargin{:});
basepath = p.Results.basepath;
sample_rate_LFP = p.Results.sample_rate_LFP;
num_samples = p.Results.num_samples;
sample_rate = p.Results.sample_rate;
pyramidal_layer = p.Results.pyramidal_layer;
num_bins_around = p.Results.num_bins_around;
mode = p.Results.mode;
saveMat = p.Results.saveMat;

%% Define SWR timepoints
num_ripples = length(ripple_classes);
num_neurons = length(spikes.UID);
time = linspace(0,(num_samples./sample_rate_LFP),num_samples); 
ripple_peak = zeros(1,num_ripples);
lfp_peak_idx = zeros(1,num_ripples);

% To find the peak, look at the pyramidal layer 
switch mode 
    case 'swr'
        ripple_window = arrayfun(@(idx) find(ripple_timestamps(idx,1) <= time & time <= ripple_timestamps(idx,end)), 1:num_ripples, 'UniformOutput', false); 
        lfp_peak_idx = arrayfun(@(idx) computeLFPPeakIdx(idx, ripple_window, ripple_classes, LFP_filtered_hilbert, pyramidal_layer, ripple_timestamps), 1:num_ripples);
        ripple_peak = arrayfun(@(idx) (ripple_timestamps(idx,1) + lfp_peak_idx(idx)/sample_rate_LFP), 1:num_ripples);

    case 'shuffle'
        % For the shuffling control, the definition of the ripple_window is more loose, because otherwise it takes too long (~days)
        ripple_window = arrayfun(@(idx) floor(ripple_timestamps(idx,1)*sample_rate_LFP:ripple_timestamps(idx,end)*sample_rate_LFP), 1:num_ripples, 'UniformOutput', false);
        peak_time = arrayfun(@(idx) ((ripple_timestamps(idx,end) - ripple_timestamps(idx,1)) / 2), 1:num_ripples);
        ripple_peak = arrayfun(@(idx) (ripple_timestamps(idx,1) + peak_time(idx)), 1:num_ripples);
end

% The firing around ipsilateral SWRs is considered only
ripples_right = find(strcmp(ripple_classes, 'right'));
ripples_left = find(strcmp(ripple_classes, 'left'));
ripple_timestamps_right = ripple_timestamps(ripples_right,:);
ripple_timestamps_left = ripple_timestamps(ripples_left,:);
ripple_peak_right = ripple_peak(ripples_right);
ripple_peak_left = ripple_peak(ripples_left);

%% Bin firing in and around normalized SWRs
% Bin periods around the SWR events into num_bins_around on each side
bin_size = mean(ripple_timestamps(:,end)-ripple_timestamps(:,1)) / 8; % sec

for c = 1:num_neurons
    spike_times_neuron{c} = spikes.times{spikes.cluster_index == c};

    if clusterSites(c) < 65
        % right hemi
        swr_firing{c} = computeFiring(length(ripples_right), ripple_timestamps_right, ripple_peak_right, sample_rate_LFP, num_bins_around, spike_times_neuron{c}, bin_size);
    else
        % left hemi
        swr_firing{c} = computeFiring(length(ripples_left), ripple_timestamps_left, ripple_peak_left, sample_rate_LFP, num_bins_around, spike_times_neuron{c}, bin_size);
    end
    
end

%% Helper functions
function peak_idx = computeLFPPeakIdx(idx, ripple_window, ripple_classes, LFP_filtered_hilbert, pyramidal_layer, ripple_timestamps)
    if strcmp(ripple_classes(idx), 'left')
        peak_idx = find(LFP_filtered_hilbert(ripple_window{idx}, pyramidal_layer(1)) == max(LFP_filtered_hilbert(ripple_window{idx}, pyramidal_layer(1))));
    elseif strcmp(ripple_classes(idx), 'right')
        peak_idx = find(LFP_filtered_hilbert(ripple_window{idx}, pyramidal_layer(2)) == max(LFP_filtered_hilbert(ripple_window{idx}, pyramidal_layer(2))));
    end
end

function swr_firing = computeFiring(num_ripples, ripple_timestamps, ripple_peak, sample_rate_LFP, num_bins_around, spike_times_neuron, bin_size)
    % Bin periods between event start and peak, and event peak and end, into four bins
    spikes_start = zeros(num_ripples, 4);
    spikes_end = zeros(num_ripples, 4);
    firing_prob_start = zeros(1, 4);
    firing_prob_end = zeros(1, 4);

    first_spikes = zeros(num_ripples, num_bins_around);
    second_spikes = zeros(num_ripples, num_bins_around);
    first_firing_prob = zeros(1, num_bins_around);
    second_firing_prob = zeros(1, num_bins_around);

    % Get bins during the event
    edges_start = arrayfun(@(idx) (linspace(min([ripple_timestamps(idx,1) : 1/sample_rate_LFP : ripple_peak(idx)]), max([ripple_timestamps(idx,1) : 1/sample_rate_LFP : ripple_peak(idx)]), 5)), 1:num_ripples, 'UniformOutput', false);
    edges_end = arrayfun(@(idx) (linspace(min([ripple_peak(idx) : 1/sample_rate_LFP : ripple_timestamps(idx,end)]), max([ripple_peak(idx) : 1/sample_rate_LFP : ripple_timestamps(idx,end)]), 5)), 1:num_ripples, 'UniformOutput', false);

    % Get bins before and after event
    first_window = arrayfun(@(idx) ((ripple_timestamps(idx,1) - num_bins_around*bin_size) : 1/sample_rate_LFP : ripple_timestamps(idx,1)), 1:num_ripples, 'UniformOutput', false);
    second_window = arrayfun(@(idx) (ripple_timestamps(idx,end) : 1/sample_rate_LFP : (ripple_timestamps(idx,end) + num_bins_around*bin_size)), 1:num_ripples, 'UniformOutput', false);

    first_edges = arrayfun(@(idx) (linspace(min(first_window{idx}), max(first_window{idx}), num_bins_around+1)), 1:num_ripples, 'UniformOutput', false);
    second_edges = arrayfun(@(idx) (linspace(min(second_window{idx}), max(second_window{idx}), num_bins_around+1)), 1:num_ripples, 'UniformOutput', false);
        
    for r = 1:num_ripples
        spikes_start(r,:) = histcounts(spike_times_neuron, edges_start{r});
        spikes_end(r,:) = histcounts(spike_times_neuron, edges_end{r});

        first_spikes(r,:) = histcounts(spike_times_neuron, first_edges{r});
        second_spikes(r,:) = histcounts(spike_times_neuron, second_edges{r});
    end

    % Firing probability
    firing_prob_start(:) = sum(spikes_start(:,:),1) ./ length(spike_times_neuron);
    firing_prob_end(:) = sum(spikes_end(:,:),1) ./ length(spike_times_neuron);

    first_firing_prob(:) = sum(first_spikes(:,:),1) ./ length(spike_times_neuron);
    second_firing_prob(:) = sum(second_spikes(:,:),1) ./ length(spike_times_neuron);

    % Output results in struct
    swr_firing.firing_prob_start = firing_prob_start; % start of SWR
    swr_firing.firing_prob_end = firing_prob_end; % end of SWR
    swr_firing.first_firing_prob = first_firing_prob; % window before SWR start
    swr_firing.second_firing_prob = second_firing_prob; % window after SWR end
    swr_firing.spikes_start = spikes_start; % start of SWR
    swr_firing.spikes_end = spikes_end; % end of SWR
    swr_firing.first_spikes = first_spikes; % window before SWR start
    swr_firing.second_spikes = second_spikes; % window after SWR end
    swr_firing.around_bin_size = bin_size; % bin size around SWR
end

end