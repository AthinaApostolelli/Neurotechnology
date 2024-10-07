function [new_theta_windows] = postprocessThetaSegments(theta_windows, LFPFromSelectedChannels, thetaLFP, thetaPhase, varargin)

% Fine tunes theta segments to only select windows that are long enough and 
% where theta is strong enough too. 
%
% INPUTS
% - num_samples             number of data points for each channel
% - sample_rate             ephys sampling rate
% - phase_threshold         threshold for detecting incorrect peaks in theta phase (deg)
% - amplitude_threshold     threshold for detecting low amplitude peaks in theta LFP (uV)
% - duration_thershold      minimum duration of theta window (s)      
%
% OUTPUT
% new_theta_windows:    N x 2 array [N theta windows x (start, end)] (samples)
%
% HISTORY
% Athina Apostolelli 2024

%% Set input parameters
p=inputParser;
addParameter(p,'basepath',pwd,@isstr);
addParameter(p,'num_samples',[],@isnumeric);
addParameter(p,'sample_rate',20000,@isnumeric);
addParameter(p,'sel_channel_idx',1,@isnumeric);
addParameter(p,'phase_threshold',15,@isnumeric);
addParameter(p,'amplitude_threshold',65,@isnumeric);
addParameter(p,'duration_threshold',0.5,@isnumeric);
addParameter(p,'cycle_threshold_buffer',0.01,@isnumeric);
addParameter(p,'plotWin',false,@islogical);
addParameter(p,'f_theta',[6,10],@isnumeric);

parse(p,varargin{:});
basepath = p.Results.basepath;
num_samples = p.Results.num_samples;
sample_rate = p.Results.sample_rate;
sel_channel_idx = p.Results.sel_channel_idx;
phase_threshold = p.Results.phase_threshold;
amplitude_threshold = p.Results.amplitude_threshold;
duration_threshold = p.Results.duration_threshold;
cycle_threshold_buffer = p.Results.cycle_threshold_buffer;
plotWin = p.Results.plotWin;
f_theta = p.Results.f_theta;

%% Get good theta windows
Time_vector_LFP = linspace(0,num_samples./sample_rate,num_samples);

% Apply phase and amplitude thresholds
theta_windows_v1 = [];
for w = 1:length(theta_windows);
    theta_window_idx = find(theta_windows(w,1) < Time_vector_LFP & Time_vector_LFP < theta_windows(w,2));
    [~,idx_peak] = findpeaks(thetaLFP(theta_window_idx, sel_channel_idx));
    mean_peakPhase = mean(thetaPhase(theta_window_idx(idx_peak), sel_channel_idx));

    start_idx = theta_window_idx(idx_peak(1));

    break_detected = 0;
    count = 0;
    for i = 2:length(idx_peak)-1
        if (thetaPhase(theta_window_idx(idx_peak(i))) > mean_peakPhase + phase_threshold) | ...
                (thetaPhase(theta_window_idx(idx_peak(i))) < mean_peakPhase - phase_threshold) | ...
                (thetaLFP(theta_window_idx(idx_peak(i))) < amplitude_threshold) 
                
            break_detected = 1;
            if count > 2
                end_idx = theta_window_idx(idx_peak(i-1));
                if start_idx < end_idx
                    theta_windows_v1 = [theta_windows_v1; start_idx, end_idx];
                    start_idx = theta_window_idx(idx_peak(i+1));
                    count = 0;
                end
            elseif count <= 2
                start_idx = theta_window_idx(idx_peak(i+1));
            end
        end
        count = count + 1;
    end
    if break_detected == 0
        theta_windows_v1 = [theta_windows_v1; theta_window_idx(1), theta_window_idx(end)];
    end
end

% Apply cycle threshold 
cycle_threshold(1) = 1 / f_theta(1) + cycle_threshold_buffer;
cycle_threshold(2) = 1 / f_theta(2) - cycle_threshold_buffer;

theta_windows_v2 = [];
for w = 1:length(theta_windows_v1)
    theta_window_idx = find(theta_windows_v1(w,1)./sample_rate < Time_vector_LFP & Time_vector_LFP < theta_windows_v1(w,2)./sample_rate);
    idx_peak_phase = find(abs(diff(thetaPhase(theta_window_idx))) > phase_threshold) - 1; % go 1 idx back
    
    if idx_peak_phase 
        start_idx = theta_window_idx(idx_peak_phase(1));
    else
        continue
    end

    break_detected = 0;
    count = 0;
    for i = 2:length(idx_peak_phase)-1
        if (idx_peak_phase(i) / sample_rate - idx_peak_phase(i-1) / sample_rate) > cycle_threshold(1) | ...
                (idx_peak_phase(i) / sample_rate - idx_peak_phase(i-1) / sample_rate) < cycle_threshold(2)
                            
            break_detected = 1;
            if count > 2
                end_idx = theta_window_idx(idx_peak_phase(i-1));
                if start_idx < end_idx
                    theta_windows_v2 = [theta_windows_v2; start_idx, end_idx];
                    start_idx = theta_window_idx(idx_peak_phase(i+1));
                    count = 0;
                end
            elseif count <= 2
                start_idx = theta_window_idx(idx_peak_phase(i+1));
            end
        end
        count = count + 1;
    end
    if break_detected == 0
        theta_windows_v2 = [theta_windows_v2; theta_window_idx(1), theta_window_idx(end)];
    end
end

% Apply duration threshold (remove short theta windows)
for n = 1:length(theta_windows_v2)
    if (theta_windows_v2(n,2) - theta_windows_v2(n,1))/sample_rate < duration_threshold
        theta_windows_v2(n,:) = nan;
    end
end
theta_windows_v2(any(isnan(theta_windows_v2),2),:) = [];
new_theta_windows = theta_windows_v2;

% Plot each theta segment (optional)
if plotWin
    test_theta(num_samples, sample_rate, LFPFromSelectedChannels(sel_channel_idx,:), new_theta_windows, thetaLFP, thetaPhase);
end

end