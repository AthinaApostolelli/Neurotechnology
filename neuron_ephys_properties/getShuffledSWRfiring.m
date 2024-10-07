function [shuffled_swr_firing] = getShuffledSWRfiring(ripple_timestamps, ...
    ripple_classes, ripple_durations, spikes, LFP_filtered_hilbert, event_range, valid_range, ...
    num_shuffles, num_ripples, clusterSites, varargin)

%% Set input parameters
p=inputParser;
addParameter(p,'basepath',pwd,@isstr);
addParameter(p,'num_samples',[],@isnumeric);
addParameter(p,'sample_rate',20000,@isnumeric);
addParameter(p,'sample_rate_LFP',2000,@isnumeric);
addParameter(p,'pyramidal_layer',[],@isnumeric);
addParameter(p,'num_bins_around',10,@isnumeric);
addParameter(p,'mode','shuffle',@isstr); % shuffle or swr
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

%% Perform the shuffles
num_neurons = length(spikes.UID);
for c = 1:num_neurons
    % Store the required data into a new struct
    clear spike_data 
    spike_data.times = {spikes.times{spikes.cluster_index == c}};
    spike_data.UID = spikes.UID(c);
    spike_data.cluster_index = 1; % artificially set to 1 so that indexing in the following function works

    shuffled_ripple_durations = zeros(num_shuffles, num_ripples);
    shuffled_ripple_timestamps = zeros(num_shuffles, num_ripples, 2);
    shuffled_ripple_classes = cell(num_shuffles, num_ripples);
    random_start_time = zeros(num_shuffles, num_ripples);
    random_end_time = zeros(num_shuffles, num_ripples);

    idx_perm = arrayfun(@(~) randperm(num_ripples), 1:num_shuffles, 'UniformOutput', false);
    idx_perm = reshape(cell2mat(idx_perm),[num_ripples,num_shuffles]);

    for p = 1:num_shuffles
        % clear shuffled_ripple_durations shuffled_ripple_classes random_start_time random_end_time event_mask
        % Shuffle the ripples, but ensure that the classes remain the same
        shuffled_ripple_durations(p,:) = ripple_durations(idx_perm(:,p));
        shuffled_ripple_classes(p,:) = ripple_classes(idx_perm(:,p));

        % Generate random start time points within the valid range
        random_start_time(p,:) = valid_range(randperm(length(valid_range), num_ripples));  % in sec
        random_end_time(p,:) = random_start_time(p,:) + shuffled_ripple_durations(p,:);
        
        % Ensure the new random event does not fall within the theta/SWR time windows.
        event_mask = ismember(random_end_time(p,:), event_range) | ismember(random_start_time(p,:), event_range);
        while any(event_mask)
            random_start_time(p,:) = valid_range(randperm(length(valid_range), num_ripples));
            random_end_time(p,:) = random_start_time(p,:) + shuffled_ripple_durations(p,:)';
            % Resample if any new event falls in the invalid range
            event_mask = ismember(random_end_times(p,:), event_range) | ismember(random_start_times(p,:), event_range);
        end
        shuffled_ripple_timestamps(p,:,:) = [random_start_time(p,:)', random_end_time(p,:)'];
    end

    %% Get the shuffled correlograms
    shuffled_neuron_swr_firing = arrayfun(@(idx) getSWRfiring(squeeze(shuffled_ripple_timestamps(idx,:,:)), shuffled_ripple_classes(idx,:), spike_data, LFP_filtered_hilbert, clusterSites, ...
        'num_samples',num_samples, 'pyramidal_layer',pyramidal_layer, 'mode',mode), 1:num_shuffles);%, 'UniformOutput', false);

    shuffled_swr_firing{c} = shuffled_neuron_swr_firing;
end
end