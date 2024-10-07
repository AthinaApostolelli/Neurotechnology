function [waveforms] = getWaveformsNew(varargin)

% INPUT
% filtWaveforms - n_samples x n_channels x n_spikes
%
% Written by Peter Gombkoto Ph.D ETH - Neurotechnology Group - pgombkoto@ethz.ch
% Adapted by Athina Apostolelli - aapostolelli@ethz.ch

%% Set parameters
p=inputParser;
addParameter(p,'basepath',pwd,@isstr);
addParameter(p,'sample_rate',20000,@isnumeric);
addParameter(p,'event_window',[-0.25, 1.5],@isnumeric);
addParameter(p,'saveMat',true,@islogical); 
addParameter(p,'forceReload',false,@islogical);
addParameter(p,'num_neurons',[],@isnumeric);

parse(p,varargin{:});
basepath = p.Results.basepath;
sample_rate = p.Results.sample_rate;
event_window = p.Results.event_window;
saveMat = p.Results.saveMat;
forceReload = p.Results.forceReload;
num_neurons = p.Results.num_neurons;

%% Load data
if ~isempty(dir(fullfile(basepath, '*.waveforms.cellinfo.mat'))) & ~forceReload
    file = dir(fullfile(basepath, '*.waveforms.cellinfo.mat'));
    load(file(1).name);
    disp('Waveforms already detected. Loading...');
    return
end

% Filtered waveforms
if contains(basepath,'concatenated')
    if ~isempty(fullfile(basepath, 'concat_data_filt.jrc'))
        fid=fopen(fullfile(basepath, 'concat_data_filt.jrc'),'r');
        filtWaveforms = fread(fid,Inf,'int16');   % spike waveform filtered by JRClust raw data from binary
        fclose(fid);
    else
        error('No filtered waveforms found in this directory');
    end
else
    if ~isempty(fullfile(basepath, 'amplifier_filt.jrc'))
        fid=fopen(fullfile(basepath, 'amplifier_filt.jrc'),'r');
        filtWaveforms = fread(fid,Inf,'int16');   % spike waveform filtered by JRClust raw data from binary
        fclose(fid);
    else
        error('No filtered waveforms found in this directory');
    end
end

% Electrode data
if isempty(sample_rate)
    [rawRecordings,sampleRate,bitScaling,shankMap,siteMap,siteLoc,probePad] = read_prb_file; % read paramater about the electrode, and raw data file.
    sample_rate = sampleRate;
end

% Spikes
if isempty(num_neurons)
    spikes = loadSpikes_JRC('basepath',basepath,'forceReload',false);
    num_neurons = len(spikes.UID);
end

% Load auxiliary data
if contains(basepath, 'concatenated')
    load(fullfile(basepath, 'concat_data_res.mat'), 'filtShape');   
    load(fullfile(basepath, 'concat_data_res.mat'), 'spikeClusters');
else
    load(fullfile(basepath, 'amplifier_res.mat'), 'filtShape');   
    load(fullfile(basepath, 'amplifier_res.mat'), 'spikeClusters');
end

n_samples = filtShape(1);
n_channels = filtShape(2);
n_spikes = filtShape(3);

spikesFilt = reshape(filtWaveforms, n_samples, n_channels, n_spikes) .*(37.4./192); % [n_samples, n_channels, n_spikes]

% time vectors, and limitations
time_vector = (linspace(0, (n_samples-1)./sample_rate, n_samples.*2)).*1000;
time_vector_original = (linspace(0, (n_samples-1)./sample_rate, n_samples)).*1000;
peak_zero_limit = find(round(time_vector,1)==0.6, 1, 'first');


%% Declare memory for variables.
PeakOne = zeros(num_neurons,3);
PeakMinusOne = zeros(num_neurons,3);
PeakZero = zeros(num_neurons,3);
PeakMinusToPeakZero = zeros(num_neurons,3);
for_PCA = zeros(num_neurons,floor(0.0002.*(2*sample_rate)));
wave = zeros(num_neurons,n_samples.*2);
std_wave = zeros(num_neurons,n_samples.*2);
PeakMinusToPeakZero = zeros(num_neurons,1);
PeakZero_PeakOne = zeros(num_neurons,1);
mean_waveform = zeros(num_neurons,n_samples);
std_waveform = zeros(num_neurons,n_samples);
norm_waveform = zeros(num_neurons,n_samples.*2);
stop = [];
peakA = zeros(num_neurons,1);
peakB = zeros(num_neurons,1);

%% Calculate peaks and times
% Note, in another version of this function, we can look at each spike
% separately.

for c = 1:num_neurons

    mean_waveform(c,:) = mean(squeeze(spikesFilt(:,1,spikeClusters==c)),2); % mean across 11 channels 
    std_waveform(c,:) = std(squeeze(spikesFilt(:,1,spikeClusters==c)),0,2); % std across 11 channels

    try       
        wave(c,:) = interp1([1:size(mean_waveform(c,:),2)],mean_waveform(c,:),[1:0.5:size(mean_waveform(c,:),2),size(mean_waveform(c,:),2)],'spline');
        std_wave(c,:) = interp1([1:size(std_waveform(c,:),2)],std_waveform(c,:),[1:0.5:size(std_waveform(c,:),2),size(std_waveform(c,:),2)],'spline');
        % wave(c,:) = interp1([1:size(mean_waveform(c,:),2)],zscore(mean_waveform(c,:)),[1:0.5:size(mean_waveform(c,:),2),size(mean_waveform(c,:),2)],'spline');
        % std_wave(c,:) = interp1([1:size(std_waveform(c,:),2)],zscore(std_waveform(c,:)),[1:0.5:size(std_waveform(c,:),2),size(std_waveform(c,:),2)],'spline');
        
        % (time, amplitude, index)
        idx_PeakZero = find(min(wave(c,1:peak_zero_limit))==wave(c,:),1,'first');
        PeakZero(c,1:3)=[time_vector(idx_PeakZero), wave(c,idx_PeakZero), idx_PeakZero];
            
        Ypk = [];
        Xpk = [];
        [Ypk,Xpk,~,~,~,~,~,~,~] = findpeaks_peti(wave(c,:),time_vector,'WidthReference','halfheight');
        [~,idx] = min(abs(Xpk-PeakZero(c,1)));
        minVal = Xpk(idx);
        PeakMinusOne(c,1:3) = [minVal, wave(c,find(time_vector==minVal,1,'first')), find(time_vector==minVal,1,'first')];

        idx_PeakOne = find(max(wave(c,PeakZero(c,3):end))==wave(c,:),1,'first');
        PeakOne(c,1:3) = [time_vector(idx_PeakOne), wave(c,idx_PeakOne), idx_PeakOne];
    
        PeakMinusToPeakZero(c,:) = diff([PeakMinusOne(c,1), PeakZero(c,1)]);
        
        peakA(c,:) = abs(diff([wave(c,PeakMinusOne(c,3)),wave(c,PeakZero(c,3))]));
        peakB(c,:) = abs(diff([wave(c,PeakZero(c,3)),wave(c,PeakOne(c,3))]));
        
        PeakZero_PeakOne(c,:) = diff([PeakZero(c,1),PeakOne(c,1)]);
        
        norm_waveform = wave(c,:)./abs(min(wave(c,1:peak_zero_limit)));
        stop = PeakZero(c,3) + floor(0.0002.*(2*sample_rate));
        for_PCA(c,:) = diff(norm_waveform(PeakZero(c,3):stop));
   
    catch
        disp('Error! Peaks could not be found.')
    end


    %% Clean up and save results
    % If waveform (P-1) starts with 0 -> out of analysis, extreme values out of analysis
    
    % if first peak is not properly detected
    if PeakMinusToPeakZero < 0
        array_max = islocalmax(wave(c,1:PeakZero(c,3)));
        if any(array_max)
            idx_max = find(array_max==1,1,'last');
        else
            idx_max = 1;
        end
        minVal1 = time_vector(idx_max);
    
        PeakMinusOne(c,1:3)=[minVal1, wave(c,find(time_vector==minVal1,1,'first')), (find(time_vector==minVal1,1,'first'))];
        PeakOne(c,1:3)=[minVal, wave(c,find(time_vector==minVal,1,'first')), (find(time_vector==minVal,1,'first'))];
    end
    
    % if second peak is too far away 
    if abs(PeakZero_PeakOne(c)) > 1
        [Ypk,Xpk,~,~,~,~,~,~,~] = findpeaks_peti(wave(c,PeakZero(c,3):end),time_vector(PeakZero(c,3):end),'WidthReference','halfheight');
        [~,idx] = min(abs(Xpk-PeakZero(c,1)));
        minVal = Xpk(idx);
        if ~isempty(minVal)
            PeakOne(1,1:3) = [minVal, wave(c,find(time_vector==minVal,1,'first')), (find(time_vector==minVal,1,'first'))];
        end
    end
end

waveforms.PeaktoTrough = PeakMinusToPeakZero;
waveforms.TroughtoPeak = PeakZero_PeakOne;
waveforms.AB_ratio = ((peakB-peakA)./(peakA+peakB))';
waveforms.Peaks = [PeakMinusOne(:,1) PeakZero(:,1) PeakOne(:,1)];
waveforms.Amp =[PeakMinusOne(:,2) PeakZero(:,2) PeakOne(:,2)];
waveforms.for_PCA = for_PCA(:,:);
waveforms.time = time_vector;
waveforms.time_original = time_vector_original;
waveforms.wave = wave; % upsampled
waveforms.std_wave = std_wave; % upsampled
waveforms.mean_waveform = mean_waveform;
waveforms.std_waveform = std_waveform;

% %% Plot results
% figure(1)
% subplot(3,3,[1 2 4 5 7 8])
% % plot(x_old.*1000,(rawWaveforms(:,:)).*6,'LineWidth',1)
% plot(time_vector, wave,'LineWidth',1)
% hold on; scatter(waveforms.Peaks(:,1),waveforms.Amp(:,1),'*');
% hold on; scatter(waveforms.Peaks(:,2),waveforms.Amp(:,2),'*');
% hold on; scatter(waveforms.Peaks(:,3),waveforms.Amp(:,3),'*');
% 
% close all

[~,basename] = fileparts(basepath);
if saveMat
    save(fullfile(basepath, [basename '.waveforms.cellinfo.mat']),'waveforms'); 
end

end

