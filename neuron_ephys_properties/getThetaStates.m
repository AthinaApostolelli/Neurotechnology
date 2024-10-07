function [theta_windows] = getThetaStates(LFPFromSelectedChannels, varargin)

% Detects windows with dominant theta power (compared to delta power). 
%
% INPUTS
% - freqlist                range of frequencies to compute the spectrogram
% - window                  widnow (s) to consider for the spectrogram
% - noverlap                overlap in the windows (s)
% - num_samples             number of data points for each channel
% - sample_rate             ephys sampling rate
% - smoothfact              number of samples used for smoothing (not used)
% - f_theta                 theta frequency band
% - f_delta                 delta frequency band
% - the2d_ratio_threshold   theta / delta ratio for theta 'state' detection
%
% OUTPUT
% theta_windows:    N x 2 array [N theta windows x (start, end)] (samples)
%
% HISTORY
% Athina Apostolelli 2024
% Adapted from buzcode: detectStates/SleepScoreMaster/ClusterStates_GetMetrics.m

%% Set input parameters
p=inputParser;
addParameter(p,'basepath',pwd,@isstr);
addParameter(p,'freqlist',[2,20],@isnumeric);
addParameter(p,'window',2,@isnumeric);
addParameter(p,'noverlap',1,@isnumeric);
addParameter(p,'num_samples',[],@isnumeric);
addParameter(p,'sample_rate',20000,@isnumeric);
addParameter(p,'smoothfact',15,@isnumeric);
addParameter(p,'f_theta',[6 10],@isnumeric);
addParameter(p,'f_delta',[2 3],@isnumeric);
addParameter(p,'th2d_ratio_threshold',1.5,@isnumeric);
addParameter(p,'saveMat',true,@islogical); 

parse(p,varargin{:});
basepath = p.Results.basepath;
freqlist = p.Results.freqlist;
window = p.Results.window;
noverlap = p.Results.noverlap;
num_samples = p.Results.num_samples;
sample_rate = p.Results.sample_rate;
smoothfact = p.Results.smoothfact;
f_theta = p.Results.f_theta;
f_delta = p.Results.f_delta;
th2d_ratio_threshold = p.Results.th2d_ratio_threshold;
saveMat = p.Results.saveMat;


%% Calculate spectrogram
freqlist = logspace(log10(freqlist(1)),log10(freqlist(2)),100);

[thFFTspec,thFFTfreqs,t_thclu] = spectrogram(single(LFPFromSelectedChannels),window*sample_rate,noverlap*sample_rate,freqlist,sample_rate);

timestamps = linspace(0, num_samples/sample_rate, num_samples);

t_thclu = t_thclu+timestamps(1); % Offset for scoretime start
specdt = mode(diff(t_thclu));
thFFTspec = (abs(thFFTspec));

% Find transients for calculating TH
% zFFTspec = NormToInt(log10(thFFTspec)','modZ');
% totz = NormToInt(abs(sum(zFFTspec,2)),'modZ');
% badtimes_TH = find(totz>3);

%% Find power of theta and delta
% Theta power
thfreqs = (thFFTfreqs>=f_theta(1) & thFFTfreqs<=f_theta(2));
thpower = sum((thFFTspec(thfreqs,:)),1);
allpower = sum((thFFTspec),1);

thratio = thpower./allpower;    % Narrowband Theta
% thratio(badtimes_TH) = nan;
% thratio = smooth(thratio,smoothfact./specdt);

% Delta power
dfreqs = (thFFTfreqs>=f_delta(1) & thFFTfreqs<=f_delta(2));
dpower = sum((thFFTspec(dfreqs,:)),1);

dratio = dpower./allpower;
% dratio(badtimes_TH) = nan;
% dratio = smooth(dratio,smoothfact./specdt);

% Moments of theta are based on the ratio of theta to delta power
th2d_ratio = thpower./dpower;
theta_moments = th2d_ratio > th2d_ratio_threshold;

%% Define theta epochs
% Theta epochs need to span at least 3 1s windows
counts = 0;
theta_segments = [];
for i = 1:length(theta_moments)
    if theta_moments(i)
        counts = counts + 1;
        if counts == 3
            theta_segments = [theta_segments, i-2, i-1, i];
        elseif counts > 3
            theta_segments = [theta_segments, i];
        end
    else
        counts = 0;
        continue
    end
end

idx_seg = find(diff(theta_segments)~=1);
theta_windows = zeros(length(idx_seg)+1, 2);

theta_windows(1,1) = theta_segments(1);
theta_windows(1,2) = theta_segments(idx_seg(1));
for i = 2:(length(idx_seg))
    theta_windows(i,1) = theta_segments(idx_seg(i-1)+1);
    theta_windows(i,2) = theta_segments(idx_seg(i));
end
theta_windows(end,1) = theta_segments(idx_seg(end)+1);
theta_windows(end,2) = theta_segments(end);

end