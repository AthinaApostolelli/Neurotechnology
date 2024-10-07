function [neuron_class_bySWR] = classifyNeurons_SWRfiring(swr_firing, shuffled_swr_firing, num_neurons, num_shuffles, varargin)

% single-peak: if the number of spikes in six bins surrounding the ripple 
% maximum is higher in the observed SWR correlogram than the mean + 2 s.d. 
% from 100 shuffled correlograms.  
% anti-sharp-wave: if the number of spikes in six bins surrounding the 
% ripple maximum is less in the observed SWR correlogram than the mean 
% + 2 s.d. from 100 shuffled correlograms. 
% biphasic: if the number of spikes in four bins surrounding the beginning 
% of the sharp wave is higher in the observed SWR correlogram than the mean 
% + 1 s.d. from 100 shuffled correlograms and the number of spikes in 10 
% consecutive bins, starting from the fourth bin of the SWR, was lower in 
% the observed SWR correlogram than the mean - 2 s.d. from the respective 
% bins from 100 shuffled correlograms. 
%
% Written by Athina Apostolelli - aapostolelli@ethz.ch

%% Set input parameters
p=inputParser;
addParameter(p,'basepath',pwd,@isstr);
addParameter(p,'class1_win_start',2,@isnumeric);
addParameter(p,'class1_win_dur',6,@isnumeric);
addParameter(p,'class2_win1_start',8,@isnumeric);
addParameter(p,'class2_win1_dur',4,@isnumeric);
addParameter(p,'class2_win2_start',14,@isnumeric);
addParameter(p,'class2_win2_dur',10,@isnumeric); 
addParameter(p,'saveMat',true,@islogical); % TODO

parse(p,varargin{:});
basepath = p.Results.basepath;
class1_win_start = p.Results.class1_win_start;
class1_win_dur = p.Results.class1_win_dur;
class2_win1_start = p.Results.class2_win1_start;
class2_win1_dur = p.Results.class2_win1_dur;
class2_win2_start = p.Results.class2_win2_start;
class2_win2_dur = p.Results.class2_win2_dur;
saveMat = p.Results.saveMat;

%% Get mean firing in each bin
single_peak_swr = zeros(num_neurons,1);
biphasic_swr = zeros(num_neurons,1);
anti_sharp_wave_swr = zeros(num_neurons,1);

for c = 1:num_neurons

    %% Concatenate windows during the events
    swr_spikes{c} = cat(2, swr_firing{c}.spikes_start(:,:), swr_firing{c}.spikes_end(:,:));

    shuffled_swr_spikes = arrayfun(@(idx) [shuffled_swr_firing{c}{idx}.spikes_start, shuffled_swr_firing{c}{idx}.spikes_end], ...
        1:num_shuffles, 'UniformOutput', false);

    % Mean across ripples (during event)
    mean_shuffled_spikes_ripples = arrayfun(@(idx) mean(shuffled_swr_spikes{idx},1), 1:num_shuffles, 'UniformOutput', false);
    mean_shuffled_spikes_array = reshape(cell2mat(mean_shuffled_spikes_ripples),[8, num_shuffles]); % 8 bins
    
    mean_spikes_ripples = mean(swr_spikes{c},1);

    % Mean across shuffles (during event)
    mean_spikes_shuffled = mean(mean_shuffled_spikes_array,2)';
    std_spikes_shuffled = std(mean_shuffled_spikes_array,[],2)'; 

    %% Concatenate windows around and during the events 
    all_spikes{c} = cat(2, swr_firing{c}.first_spikes(:,:), swr_firing{c}.spikes_start(:,:), swr_firing{c}.spikes_end(:,:), ...
        swr_firing{c}.second_spikes(:,:));

    shuffled_all_spikes = arrayfun(@(idx) [shuffled_swr_firing{c}{idx}.first_spikes, shuffled_swr_firing{c}{idx}.spikes_start, ...
        shuffled_swr_firing{c}{idx}.spikes_end, shuffled_swr_firing{c}{idx}.second_spikes], ...
        1:num_shuffles, 'UniformOutput', false);

    % Mean across ripples (around and during event)
    mean_shuffled_spikes_all = arrayfun(@(idx) mean(shuffled_all_spikes{idx},1), 1:num_shuffles, 'UniformOutput', false);
    mean_shuffled_spikes_all_array = reshape(cell2mat(mean_shuffled_spikes_all),[28, num_shuffles]); % 10 + 8 + 10 = 28 bins

    mean_spikes_all = mean(all_spikes{c},1);

    % Mean across shuffles (around and during event)
    mean_spikes_shuffled_all = mean(mean_shuffled_spikes_all_array,2)';
    std_spikes_shuffled_all = std(mean_shuffled_spikes_all_array,[],2)'; 

    % Optionally plot histograms. The shuffled histograms should be homogeneous
    % figure;
    % h1 = bar(1:8, mean_spikes_ripples, 'hist');
    % set(h1, 'FaceColor', 'r', 'EdgeColor', 'white', 'FaceAlpha', 0.5);
    % hold on
    % h2 = bar(1:8, mean_spikes_shuffled, 'hist');
    % set(h2, 'FaceColor', 'b', 'EdgeColor', 'white', 'FaceAlpha', 0.5);
    % ylabel('Mean number of spikes')

    %% Classify neuron based on SWR firing 
    
    % Classify single-peak neurons
    count = 0;
    for b = class1_win_start:class1_win_start+class1_win_dur  % look at 6 bins 
        if mean_spikes_ripples(b) > (mean_spikes_shuffled(b) + 2*std_spikes_shuffled(b))
            count = count + 1;
            if count == 6
                single_peak_swr(c) = 1;
            else
                continue
            end
        else
            break 
        end
    end

    % Classify anti-sharp-wave neurons
    count = 0;
    for b = class1_win_start:class1_win_start+class1_win_dur  % look at 6 bins 
        if mean_spikes_ripples(b) < (mean_spikes_shuffled(b) + 2*std_spikes_shuffled(b))
            count = count + 1;
            if count == 6
                anti_sharp_wave_swr(c) = 1;
            else
                continue
            end
        else
            break 
        end
    end

    % Classify biphasic neurons 
    count1 = 0;
    count2 = 0;
    first_condition_met = 0;
    for b = class2_win1_start:class2_win1_start+class2_win1_dur
        if mean_spikes_all(b) > (mean_spikes_shuffled_all(b) + std_spikes_shuffled_all(b))
            count1 = count1 + 1;
            if count1 == 4 
                first_condition_met = 1;
            end
        end
    end
    if first_condition_met
        for b = class2_win2_start:class2_win2_start+class2_win2_dur
            if mean_spikes_all(b) < (mean_spikes_shuffled_all(b) - 2*std_spikes_shuffled_all(b))
                count2 = count2 + 1;
                if count2 == 10
                    biphasic_swr(c) = 1;
                end
            end
        end
    end
end

%% Output results in struct
neuron_class_bySWR.mean_spikes_swr = mean_spikes_all;
neuron_class_bySWR.mean_spikes_shuffled = mean_spikes_shuffled_all;
neuron_class_bySWR.std_spikes_shuffled = std_spikes_shuffled_all;
neuron_class_bySWR.single_peak_swr = single_peak_swr;
neuron_class_bySWR.biphasic_swr = biphasic_swr;
neuron_class_bySWR.anti_sharp_wave_swr = anti_sharp_wave_swr;

end