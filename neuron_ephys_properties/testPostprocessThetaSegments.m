function [new_theta_windows] = testPostprocessThetaSegments(theta_windows, LFPFromSelectedChannels, thetaLFP, thetaPhase, varargin)

% Fine tunes theta segments to only select windows that are long enough and 
% where theta is strong enough too. 
%
% INPUTS
% - num_samples             number of data points for each channel
% - sample_rate             ephys sampling rate
% - phase_threshold         threshold for detecting incorrect peaks in theta phase (deg)
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
addParameter(p,'phase_threshold',15,@isnumeric);
addParameter(p,'amplitude_threshold',65,@isnumeric);
addParameter(p,'duration_threshold',0.5,@isnumeric);

parse(p,varargin{:});
basepath = p.Results.basepath;
num_samples = p.Results.num_samples;
sample_rate = p.Results.sample_rate;
phase_threshold = p.Results.phase_threshold;
amplitude_threshold = p.Results.amplitude_threshold;
duration_threshold = p.Results.duration_threshold;

%% Get good theta windows
Time_vector_LFP = linspace(0,num_samples./sample_rate,length(LFPFromSelectedChannels));

% Apply phase and amplitude thresholds

new_theta_windows = [];
for w = 1:length(theta_windows);
    theta_window_idx = find(theta_windows(w,1)./sample_rate < Time_vector_LFP & Time_vector_LFP < theta_windows(w,2)./sample_rate);
    [~,idx_peak] = findpeaks(thetaLFP(theta_window_idx,1));

    start_idx = theta_window_idx(idx_peak(1));

    break_detected = 0;
    count = 0;
    peaks = [];
    for j = 1:length(idx_peak)
        peaks = [peaks, thetaLFP(theta_window_idx(idx_peak(j)))];
    end
    
    for i = 2:length(idx_peak)-1
        if (thetaLFP(theta_window_idx(idx_peak(i))) < amplitude_threshold)

            break_detected = 1;
            if count > 2
                end_idx = theta_window_idx(idx_peak(i-1));
                if start_idx < end_idx
                    new_theta_windows = [new_theta_windows; start_idx, end_idx];
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
        new_theta_windows = [new_theta_windows; theta_window_idx(1), theta_window_idx(end)];
    end
end


% Apply duration threshold (remove short theta windows)
for n = 1:length(new_theta_windows)
    if (new_theta_windows(n,2) - new_theta_windows(n,1))/sample_rate < duration_threshold
        new_theta_windows(n,:) = nan;
    end
end
new_theta_windows(any(isnan(new_theta_windows),2),:) = [];


% Plot each theta segment (optional)
disp('Amplitude threshold.')
test_theta(num_samples, sample_rate, LFPFromSelectedChannels, new_theta_windows, thetaLFP, thetaPhase);

end