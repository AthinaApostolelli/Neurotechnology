function [theta_info] = thetaDetectionWrapper(varargin)

%% Set parameters
p=inputParser;
addParameter(p,'basepath',pwd,@isstr);
addParameter(p,'forceReload',true,@islogical);
addParameter(p,'selectedChannels',[94,38],@isnumeric);
addParameter(p,'freqlist',[2,20],@isnumeric);
addParameter(p,'t_window',2,@isnumeric); % sec
addParameter(p,'noverlap',1,@isnumeric); % sec
addParameter(p,'sample_rate',20000,@isnumeric); % Hz
addParameter(p,'bitScaling',0.195,@isnumeric); % uV
addParameter(p,'nChansInRawFile',128,@isnumeric);
addParameter(p,'sel_channel_idx',1,@isnumeric);
addParameter(p,'smoothfact',15,@isnumeric);
addParameter(p,'f_theta',[6 10],@isnumeric);
addParameter(p,'f_delta',[2 3],@isnumeric);
addParameter(p,'th2d_ratio_threshold',1.5,@isnumeric);
addParameter(p,'phase_threshold',15,@isnumeric);
addParameter(p,'duration_threshold',0.5,@isnumeric);
addParameter(p,'amplitude_threshold',60,@isnumeric); % uV
addParameter(p,'cycle_threshold_buffer',0.01,@isnumeric);
addParameter(p,'plotWin',true,@islogical);
addParameter(p,'saveMat',true,@islogical); 

parse(p,varargin{:});
basepath = p.Results.basepath;
forceReload = p.Results.forceReload;
selectedChannels = p.Results.selectedChannels;
freqlist = p.Results.freqlist;
t_window = p.Results.t_window;
noverlap = p.Results.noverlap;
sample_rate = p.Results.sample_rate;
bitScaling = p.Results.bitScaling;
nChansInRawFile = p.Results.nChansInRawFile;
sel_channel_idx = p.Results.sel_channel_idx;
smoothfact = p.Results.smoothfact;
f_theta = p.Results.f_theta;
f_delta = p.Results.f_delta;
th2d_ratio_threshold = p.Results.th2d_ratio_threshold;
phase_threshold = p.Results.phase_threshold;
duration_threshold = p.Results.duration_threshold;
amplitude_threshold = p.Results.amplitude_threshold;
cycle_threshold_buffer = p.Results.cycle_threshold_buffer;
plotWin = p.Results.plotWin;
saveMat = p.Results.saveMat;

% Check if theta info has already been calculated 
if ~isempty(fullfile(basepath, '*.theta_info_new.mat')) & ~forceReload
    disp('Theta info already determined. Loading...');
    file = dir(fullfile(basepath, '*.theta_info_new.mat'));
    load(file(1).name);
    return 
end

%% Load raw data
% Raw data info
if contains(basepath, 'concatenated')
    raw_file = fullfile(basepath, '128ch_concat_data.dat');
else
    raw_file = fullfile(basepath, 'amplifier.dat'); 
end
FileInfo = dir(raw_file);
nChansInRawFile = nChansInRawFile;
num_samples = FileInfo.bytes/(nChansInRawFile * 2); % int16 = 2 bytes

% Map the binary file into memory
RawDat_map_memory_file = memmapfile(char(raw_file), 'Format', 'int16');
totalDataPoints = numel(RawDat_map_memory_file.Data);
samplesPerChannel = totalDataPoints / nChansInRawFile;

% Initialize a matrix to hold the data from the selected channels
LFPFromSelectedChannels = zeros(numel(selectedChannels), samplesPerChannel, 'int16');

% Extract data for each selected channel
for i = 1:numel(selectedChannels)
    channelIdx = selectedChannels(i);
    LFPFromSelectedChannels(i,:) = RawDat_map_memory_file.Data(channelIdx:nChansInRawFile:end);
end
fprintf('Data loading complete.\n');

% Scalwe by the smallest voltage step the ADC can resolve (LSB) – is (2.45 V) / (216) = 37.4 μV.  
% Dividing this LSB level by the RHD2000 amplifier gain of 192
LFPFromSelectedChannels = double(LFPFromSelectedChannels).*bitScaling;  

% Helper time vector
Time_vector_LFP = linspace(0, num_samples./sample_rate, length(LFPFromSelectedChannels));

%% Get theta state according to spectrogram

% Find theta epochs based on LFP of pyramidal layer or other chosen channel
theta_windows = getThetaStates(LFPFromSelectedChannels(sel_channel_idx,:),'freqlist',freqlist,'window',t_window,'noverlap',noverlap, ...
    'num_samples',num_samples,'sample_rate',sample_rate,'f_theta',f_theta,'f_delta',f_delta,'th2d_ratio_threshold',th2d_ratio_threshold);

%% Extract theta phase
[thetaPhase, thetaLFP] = getThetaPhase(LFPFromSelectedChannels,'sample_rate',sample_rate,'f_theta',f_theta);

%% Remove segments with weak theta
new_theta_windows = postprocessThetaSegments(theta_windows, LFPFromSelectedChannels, thetaLFP, thetaPhase, 'phase_threshold',phase_threshold, ...
    'sample_rate',sample_rate,'num_samples',num_samples,'amplitude_threshold',amplitude_threshold,...
    'duration_threshold',duration_threshold,'cycle_threshold_buffer',cycle_threshold_buffer,'sel_channel_idx',sel_channel_idx,'plotWin',true,'f_theta',f_theta);

%% Save theta detection info
[~,sessname] = fileparts(basepath);
theta_info.theta_segments = new_theta_windows;
theta_info.thetaLFP = thetaLFP;
theta_info.thetaPhase = thetaPhase;
theta_info.params.f_theta = f_theta;
theta_info.params.f_delta = f_delta;
theta_info.params.th2d_ratio_threshold = th2d_ratio_threshold;
theta_info.params.window = t_window;
theta_info.params.noverlap = noverlap;
theta_info.params.phase_threshold = phase_threshold;
theta_info.params.amplitude_threshold = amplitude_threshold;
theta_info.params.cycle_threshold_buffer = cycle_threshold_buffer;
theta_info.params.duration_threshold = duration_threshold;

if saveMat
    save(fullfile(basepath, [sessname '.theta_info_new.mat']),'theta_info');
end
end