function [swr_firing] = getSWRfiring(ripple_timestamps, ripple_classes, spikes, LFP_filtered_hilbert, varargin)

%% Define SWR timepoints
% LFP_filtered_hilbert = gpuArray(LFP_filtered_hilbert);

num_ripples = length(ripple_classes);

% To find the peak, look at the pyramidal layer 
time = linspace(0,(num_samples./sample_rate_LFP),num_samples); 
ripple_start(:) = gpuArray(ripple_timestamps(:,1));
ripple_end(:) = gpuArray(ripple_timestamps(:,end)');
ripple_peak = zeros(num_ripples,1,'gpuArray');
lfp_peak_idx = zeros(num_ripples,1,'gpuArray');
    
parfor r = 1:num_ripples
    % ripple_window = find(bsxfun(@ge, ripple_start(r), time) & bsxfun(@le, ripple_end(r), time));
    % window = floor(ripple_start(r) * sample_rate_LFP) : 1 : floor(ripple_end(r) * sample_rate_LFP);
    ripple_window = find(ripple_start(r) <= time & time <= ripple_end(r));

    switch mode
        case 'swr'
            if strcmp(ripple_classes(r), 'left')
                lfp_peak_idx(r) = find(LFP_filtered_hilbert(ripple_window, pyramidal_layer(1)) == max(LFP_filtered_hilbert(ripple_window, pyramidal_layer(1))));
            elseif strcmp(ripple_classes(r), 'right')
                lfp_peak_idx(r) = find(LFP_filtered_hilbert(ripple_window, pyramidal_layer(2)) == max(LFP_filtered_hilbert(ripple_window, pyramidal_layer(2))));
            end
        case 'shuffle'
            lfp_peak_idx(r) = round((ripple_end(r)-ripple_start(r))/2);
    end

    ripple_peak(r) = ripple_start(r) + lfp_peak_idx(r) / sample_rate_LFP;
    % clear ripple_window
end

%% Bin firing in normalized SWRs
% Bin periods between event start and peak, and event peak and end, into four bins
spikes_start = gpuArray(zeros(length(spikes.UID), num_ripples, 4));
spikes_end = gpuArray(zeros(length(spikes.UID), num_ripples, 4));

firing_prob_start = gpuArray(zeros(length(spikes.UID), 4));
firing_prob_end = gpuArray(zeros(length(spikes.UID), 4));

% Bin periods around the SWR events
bin_size = mean(ripple_end-ripple_start) / 8; % sec

first_spikes = gpuArray(zeros(length(spikes.UID), num_ripples, num_bins_around));
second_spikes = gpuArray(zeros(length(spikes.UID), num_ripples, num_bins_around));

first_firing_prob = gpuArray(zeros(length(spikes.UID), num_bins_around));
second_firing_prob = gpuArray(zeros(length(spikes.UID), num_bins_around));

for c = 1:length(spikes.UID)

    spike_times_neuron{c} = spikes.times{spikes.cluster_index == c};

    parfor r = 1:num_ripples
        % Get bins during the event
        edges_start = linspace(min([ripple_start(r) : 1/sample_rate_LFP : ripple_peak(r)]), max([ripple_start(r) : 1/sample_rate_LFP : ripple_peak(r)]), 5);
        edges_end = linspace(min([ripple_peak(r) : 1/sample_rate_LFP : ripple_end(r)]), max([ripple_peak(r) : 1/sample_rate_LFP : ripple_end(r)]), 5);

        spikes_start(c,r,:) = histcounts(spike_times_neuron{c}, edges_start);
        spikes_end(c,r,:) = histcounts(spike_times_neuron{c}, edges_end);

        % Get bins before and after event
        first_window = (ripple_start(r) - num_bins_around*bin_size) : 1/sample_rate_LFP : ripple_start(r);
        second_window = ripple_end(r) : 1/sample_rate_LFP : (ripple_end(r) + num_bins_around*bin_size);

        first_edges = linspace(min(first_window), max(first_window), num_bins_around+1);
        second_edges = linspace(min(second_window), max(second_window), num_bins_around+1);

        first_spikes(c,r,:) = histcounts(spike_times_neuron{c}, first_edges);
        second_spikes(c,r,:) = histcounts(spike_times_neuron{c}, second_edges);
    end

    % Firing probability
    firing_prob_start(c,1:4) = sum(spikes_start(c,:,:),2) ./ length(spike_times_neuron{c});
    firing_prob_end(c,1:4) = sum(spikes_end(c,:,:),2) ./ length(spike_times_neuron{c});

    first_firing_prob(c,:) = sum(first_spikes(c,:,:),2) ./ length(spike_times_neuron{c});
    second_firing_prob(c,:) = sum(second_spikes(c,:,:),2) ./ length(spike_times_neuron{c});
end

%% Output results in struct
swr_firing.firing_prob_start = firing_prob_start; % start of SWR
swr_firing.firing_prob_end = firing_prob_end; % end of SWR
swr_firing.first_firing_prob = first_firing_prob; % window before SWR start
swr_firing.second_firing_prob = second_firing_prob; % window after SWR end
swr_firing.spikes_start = spikes_start; % start of SWR
swr_firing.spikes_end = spikes_end; % end of SWR

end