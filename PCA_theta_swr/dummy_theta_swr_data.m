%% Generate dummy data theta
% Phase modulation equation: f(th) = exp(A) * exp(cos(th-th_pref) - 1);
% th_pref = preferred theta phase
% A = baseline firing rate 

% Number of spikes at theta phase th: 
% p(k|lambda) = lambda * exp(-lambda) / k!
% lambda = dt * f(th)

% Parameters
fs = 20000;           % Sampling frequency in Hz
duration = 10;       % Duration of the signal in seconds
frequency = 10;       % Frequency of the sine wave in Hz
phase_bins = [0:90:270];  % Theta phases that each neuron can be tuned to
phase_hist_bins = [0:90:360];

% Generate neuron properties 
firing_rates = [0.01:0.01:20]; % possible firing rates 
log_firing_rates = log(firing_rates);
mean_fr = mean(log_firing_rates);
std_fr = std(log_firing_rates);

figure; % sanity check 
subplot(2,2,1)
plot(firing_rates, lognpdf(firing_rates, mean_fr, std_fr));
title('firing rates');
subplot(2,2,2)
plot(log_firing_rates, normpdf(log_firing_rates));
title('log firing rates');

base_fr = lognrnd(mean_fr, std_fr, [length(cells),1]); % baseline fr follows lognormal distribution
preferred_phase = datasample(phase_bins, length(cells)); 

% Generate 'LFP' data 
t = 0:1/fs:duration-1/fs; % time vector
sine_wave = sin(2*pi*frequency*t); % theta oscillation
phase = mod(angle(hilbert(sine_wave))*180/pi, 360); % theta phase 

subplot(2,2,3);
plot(t(1:fs), sine_wave(1:fs)); % 1 sec
xlabel('Time (s)');
ylabel('Amplitude');
title('Sine Wave');

% Generate spikes and calculate firing probability 
for c = 1:length(cells)
    neuron_fr = exp(base_fr(c)) * exp(cosd(phase - preferred_phase(c)));
    neuron_lambda = 1/fs * neuron_fr;
    neuron_spikes{c} = poissrnd(neuron_lambda);
    
    spike_idx = find(neuron_spikes{c});
    spike_phase = phase(spike_idx);

    binned_spikes{c} = histcounts(spike_phase, phase_hist_bins);
    firing_prob{c} = binned_spikes{c} ./ sum(neuron_spikes{c});
end


%% Generate dummy SWR data 
% Firing in excited neurons is modelled as Gaussian and firing in inhibited
% neurons is modelled as flipped bell curve. 

% Random excitation or inhibition generator: 1 = excitation, 0 = inhibition
exc_inh = randi([0,1],[length(cells),1]);

% Generate neuron properties
exc_firing_prob = [0.02:0.002:0.1]; % possible mean firing prob 
inh_firing_prob = [0.001:0.0001:0.005];

figure(2); % sanity check 
subplot(1,2,1)
plot(exc_firing_prob, normpdf(exc_firing_prob, mean(exc_firing_prob), std(exc_firing_prob)));
title('excitation firing rates');
subplot(1,2,2)
plot(inh_firing_prob, normpdf(inh_firing_prob, mean(inh_firing_prob), std(inh_firing_prob)));
title('inhibition firing rates');

exc_fr = normrnd(mean(exc_firing_prob), std(exc_firing_prob), [length(cells),1]); 
inh_fr = normrnd(mean(inh_firing_prob), std(inh_firing_prob), [length(cells),1]);




% 8 bins for each type of data (theta and SWR)
theta_data = [];
theta_data_smooth = [];
swr_data = [];
swr_shuffle_data = [];

for c = 1:length(cells)
    cell_theta_data{c} = (theta_firing.firing_prob(cells(c),:));
    f = fit([1:size(cell_theta_data{c},2)]', cell_theta_data{c}', 'smoothingspline');
    cell_theta_data_smooth{c} = f(1:size(cell_theta_data{c},2))';

    % cell_swr_data{c} = [swr_firing{cells(c)}.first_firing_prob, swr_firing{cells(c)}.firing_prob_start, swr_firing{cells(c)}.firing_prob_end, swr_firing{cells(c)}.second_firing_prob];
    % cell_swr_shuffle_data{c} = swr_firing{cells(c)}.mean_firing_prob_shuffled;
    % Keep only SWR response from start to end of event 
    cell_swr_data{c} = [swr_firing{cells(c)}.firing_prob_start, swr_firing{cells(c)}.firing_prob_end];
    cell_swr_shuffle_data{c} = [swr_firing{cells(c)}.mean_firing_prob_shuffled(11:18)];
end

theta_data = cell2mat(cell_theta_data');
theta_data_smooth = cell2mat(cell_theta_data_smooth');
% theta_data_smooth = (repmat(theta_data_smooth,[1,2])); 
swr_data = cell2mat(cell_swr_data');
swr_shuffle_data = cell2mat(cell_swr_shuffle_data');
rr_swr = swr_data./swr_shuffle_data;
rr_swr = zscore(rr_swr,0,1);

% z-scoring (first within cell and then across cells)
% zscore_theta_cell = zscore(theta_data_smooth,0,2); % dim1 = column-wise / dim2 = row-wise
zscore_theta = zscore(theta_data_smooth,0,1);

% Data used for clustering and sorting
phase = linspace(-180,180,18);
phase_new = linspace(-180,180,100);
for c = 1:size(zscore_theta,1)
    zscore_theta_interp(c,:) = interp1(phase, zscore_theta(c,:), phase_new,'spline');
    theta_data_interp(c,:) = interp1(phase, theta_data(c,:), phase_new,'spline');
end

concat_theta_swr = [zscore_theta, rr_swr];
