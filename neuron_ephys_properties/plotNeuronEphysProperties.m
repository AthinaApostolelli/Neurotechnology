function plotNeuronEphysProperties(varargin)

%% Set parameters
p = inputParser;
addParameter(p,'basepath',pwd,@isfolder);
addParameter(p,'plotpath',pwd,@ischar);
addParameter(p,'sample_rate',20000,@isnumeric);
addParameter(p,'sample_rate_LFP',2000,@isnumeric);
addParameter(p,'pathLFP',pwd,@ischar);
addParameter(p,'LFP_filtered',[],@isnumeric);
addParameter(p,'LFP_filtered_hilbert',[],@isnumeric);
addParameter(p,'LFP_channel',[],@isnumeric);
addParameter(p,'selectedChannels',[],@isnumeric);
addParameter(p,'sel_channel_idx',2,@isnumeric); 
addParameter(p,'event_window',[-0.25, 1.5],@isnumeric);
addParameter(p,'example_theta_window',50,@isnumeric);
addParameter(p,'ripple_num',1700,@isnumeric);
addParameter(p,'bin_edges_theta',[],@isnumeric);
addParameter(p,'siteMap',[],@isnumeric);
addParameter(p,'clusterSites',[],@isnumeric);
addParameter(p,'nChansInRawFile',[],@isnumeric);
addParameter(p,'nChansInLFPFile',[],@isnumeric);

parse(p,varargin{:});
basepath = p.Results.basepath;
plotpath = p.Results.plotpath;
sample_rate = p.Results.sample_rate;
sample_rate_LFP = p.Results.sample_rate_LFP;
pathLFP = p.Results.pathLFP;
LFP_filtered = p.Results.LFP_filtered;
LFP_filtered_hilbert = p.Results.LFP_filtered_hilbert;
LFP_channel = p.Results.LFP_channel;
selectedChannels = p.Results.selectedChannels;
sel_channel_idx = p.Results.sel_channel_idx;
event_window = p.Results.event_window;
example_theta_window = p.Results.example_theta_window;
ripple_num = p.Results.ripple_num;
bin_edges_theta = p.Results.bin_edges_theta;
siteMap = p.Results.siteMap;
clusterSites = p.Results.clusterSites;
nChansInRawFile = p.Results.nChansInRawFile;
nChansInLFPFile = p.Results.nChansInLFPFile;

if ~isdir(plotpath)
    mkdir(plotpath);
end

[~,basename] = fileparts(basepath);

%% Load data
% Firing rate and MRI map info
if ~isempty(dir(fullfile(basepath, '*.fr_mapElastic_info.cellinfo.mat')))
    file = dir(fullfile(basepath, '*.fr_mapElastic_info.cellinfo.mat'));
    load(file(1).name);
    disp('Firing rate and MRI map info already detected. Loading...');
end

% Waveforms
if ~isempty(dir(fullfile(basepath, '*.waveforms.cellinfo.mat')))
    file = dir(fullfile(basepath, '*.waveforms.cellinfo.mat'));
    load(file(1).name);
    disp('Waveforms already detected. Loading...');
end

% Theta info 
if ~isempty(dir(fullfile(basepath, '*.theta_info_new.mat')) )
    disp('Theta info already detected. Loading...');
    file = dir(fullfile(basepath, '*.theta_info_new.mat'));
    load(file(1).name);
end

% Theta firing 
if ~isempty(dir(fullfile(basepath, '*.theta_firing_new.cellinfo.mat')))
    disp('Theta firing already detected. Loading...');
    file = dir(fullfile(basepath, '*.theta_firing_new.cellinfo.mat'));
    load(file(1).name);
end

% Curated SWRs
if ~isempty(dir(fullfile(basepath, '*.curated_ripples.mat')))
    disp('Curated SWRs already detected. Loading...');
    file = dir(fullfile(basepath, '*.curated_ripples.mat'));
    load(file(1).name);
    num_ripples = length(ripple_classes);
else
    error('The curated ripple file (CNN detection) does not exist in this directory.');
end

% SWR firing 
if ~isempty(dir(fullfile(basepath, '*.swr_firing_new.cellinfo.mat')) )
    disp('SWR firing already detected. Loading...');
    file = dir(fullfile(basepath, '*.swr_firing_new.cellinfo.mat'));
    load(file(1).name);
end

% Raw data 
if contains(basepath, 'concatenated')
    raw_file = fullfile(basepath, '128ch_concat_data.dat');
else
    raw_file = fullfile(basepath, 'amplifier.dat'); 
end
FileInfo = dir(raw_file);
num_samples = FileInfo.bytes/(nChansInRawFile * 2); % int16 = 2 bytes

RawDat_map_memory_file = memmapfile(char(raw_file), 'Format', 'int16');
totalDataPoints = numel(RawDat_map_memory_file.Data);

LFPFromSelectedChannels = zeros(numel(selectedChannels), num_samples, 'int16');

for i = 1:numel(selectedChannels)
    channelIdx = selectedChannels(i);
    LFPFromSelectedChannels(i,:) = RawDat_map_memory_file.Data(channelIdx:nChansInRawFile:end);
end

% Spikes
spikes = loadSpikes_JRC('basepath',basepath,'forceReload',false);
num_neurons = length(spikes.UID);

% Load and filter LFP (downsampled and filtered)
if isempty(LFP_filtered) | isempty(LFP_filtered_hilbert)
    if ~isempty(pathLFP)
        FileInfo = dir(pathLFP);
        LFP = ImporterDAT_multi(fullfile(FileInfo.folder, FileInfo.name), nChansInLFPFile, [1:nChansInLFPFile]);
        
        d = designfilt('bandpassfir','FilterOrder',600,'StopbandFrequency1',125,'PassbandFrequency1',130,'PassbandFrequency2',245,'StopbandFrequency2',250,'SampleRate',2000);
        LFP_detrended = detrend(double(LFP')); % raw
        LFP_filtered = filtfilt(d, LFP_detrended); % ripples
        LFP_filtered_hilbert = abs(hilbert(LFP_filtered)); % hilbert transform
    else
        error('Please provide a valid path to the LFP file.');
    end
end
num_lfp_samples = length(LFP);

% Spiking and electrode data
if isempty(sample_rate) | isempty(siteMap)
    [rawRecordings,sampleRate,bitScaling,shankMap,siteMap,siteLoc,probePad] = read_prb_file; % read paramater about the electrode, and raw data file.
    sample_rate = sampleRate;
end

if isempty(clusterSites)
    if contains(basepath, 'concatenated')
        load(fullfile(basepath, 'concat_data_res.mat'), 'clusterSites');
    else
        load(fullfile(basepath, 'amplifier_res.mat'), 'clusterSites');
    end
end

%% Start plotting
for c = 1:num_neurons
    gcf_manual=figure();
    set(gcf_manual,'Visible','on')
    % set(gcf,'defaultTextInterpreter','none');

    %% A. Plot waveform and trough to peak info
    subplot(3,2,1)

    plot(waveforms.time, waveforms.wave(c,:), 'b');
    hold on
    jbfill(waveforms.time_original, waveforms.mean_waveform(c,:) + waveforms.std_waveform(c,:), ...
        waveforms.mean_waveform(c,:) - waveforms.std_waveform(c,:), 'b', 'none', 0, 0.3);

    xlim([min(waveforms.time), max(waveforms.time)]);
    xticks([min(waveforms.time):0.25:max(waveforms.time)])
    xticklabels([event_window(1):0.25:[event_window(2)]])
    xlabel('Time (ms)')

    % scatter(waveforms(c).Peaks(2),waveforms(c).Amp(2),'*');
    % scatter(waveforms(c).Peaks(3),waveforms(c).Amp(3),'*');

    line_x = [waveforms.Peaks(c,2); waveforms.Peaks(c,3)];
    peak_waveform = max(waveforms.mean_waveform(c,:)+waveforms.std_waveform(c,:));
    line_y = [peak_waveform + 0.2*peak_waveform, peak_waveform + 0.2*peak_waveform];
    plot(line_x, line_y, 'Color', 'k', 'LineWidth', 0.7)

    text_x = (line_x(1) + line_x(2)) / 2;
    text_y = peak_waveform + 0.35*peak_waveform;
    text(text_x, text_y, sprintf('%.3f ms', waveforms.TroughtoPeak(c)), 'HorizontalAlignment', 'center', 'FontSize', 8)

    ylabel('Amplitude (uV)')
    ylim([min(waveforms.Amp(c,:))-30, max(waveforms.Amp(c,:))+60])
    set(gca,'box','off')

    %% B. Plot autocorrelogram
    subplot(3,2,2)

    % bin_size = 0.002;
    % isi1 = diff(spikes.times{spikes.cluster_index == c});
    % isi2 = diff(flip(spikes.times{spikes.cluster_index == c})); % calculate ISI in the opposite direction too
    % bin_edges_acg = -0.2:bin_size:0.2; % 200 ms
    % bin_centers = bin_edges_acg(1:end-1) + bin_size/2;
    % isi_bins = histcounts([isi1,isi2], bin_edges_acg);
    % bar(bin_centers, isi_bins);
    % ylabel('Spike counts')
    % xlabel('Time (s)')

    % Alternative ACG plot
    [ccg,t] = CCG({spikes.times{spikes.cluster_index == c}},[],'norm','rate','duration',0.1,'binSize',0.001); % -50 to +50 ms
    b = bar(t,ccg,'hist');
    b.EdgeColor = 'none';
    ylabel('Firing rate (Hz)')
    xticks([t(1) t(end)])
    xticklabels([t(1)*1000, t(end)*1000])
    xlabel('Time (ms)')
    set(gca,'box','off')
    acg{c} = ccg;

    %% C. Plot example SWR (right hemi)
    subplot(3,2,3)
    bin_size_swr = swr_firing{1}.around_bin_size;
    assert(strcmp(ripple_classes(ripple_num),'right'), 'This SWR was in the left hemisphere. Please select one from the right.');
    time = linspace(0,(num_lfp_samples./sample_rate_LFP),num_lfp_samples);

    swr_window = find((ripple_timestamps(ripple_num,1) - 10*bin_size_swr) <= time & time <= (ripple_timestamps(ripple_num,end) + 10*bin_size_swr));
    plot(swr_window, LFP_filtered(swr_window, LFP_channel), 'DisplayName', 'filtered LFP');
    hold on
    plot(swr_window, LFP_filtered_hilbert(swr_window, LFP_channel), 'DisplayName', 'hilbert transform');
    max_idx = find(LFP_filtered_hilbert(swr_window, LFP_channel)==max(LFP_filtered_hilbert(swr_window, LFP_channel))) + swr_window(1);
    xline(max_idx, '--k', 'LineWidth', 1, 'HandleVisibility','off'); % peak
    [~,start_idx] = min(abs(time - ripple_timestamps(ripple_num,1)));
    xline(start_idx, '--k', 'LineWidth', 1, 'HandleVisibility','off'); % start
    [~,end_idx] = min(abs(time - ripple_timestamps(ripple_num,end)));
    xline(end_idx, '--k', 'LineWidth', 1, 'HandleVisibility','off'); % end
    xlim([min(swr_window), max(swr_window)]);
    xticks([start_idx, max_idx, end_idx]);
    xticklabels({'-1', '0', '1'});
    xlabel('Normalized time');
    ylim([min(LFP_filtered(swr_window, LFP_channel))-100, max(LFP_filtered(swr_window, LFP_channel))+100])
    ylabel('LFP (uV)')
    set(gca,'box','off')
    legend

    %% D. Plot example theta segment and theta phase from data (right hemi)
    subplot(3,2,4)
    theta_window = theta_info.theta_segments(example_theta_window) : (theta_info.theta_segments(example_theta_window) + sample_rate); % 1 sec window

    [troughs, troughs_idx] = findpeaks(-theta_info.thetaPhase(theta_window,sel_channel_idx)); %islocalmin(theta_info.thetaPhase(theta_window,2));
    % troughs_idx = find(troughs==1);
    [peaks, peaks_idx] = findpeaks(theta_info.thetaPhase(theta_window(troughs_idx(1):end),sel_channel_idx)); %islocalmax(theta_info.thetaPhase(theta_window(troughs_idx(1):end),2));
    % peaks_idx = find(peaks==1) + troughs_idx(1);
    theta_short_window = theta_window(troughs_idx(1):troughs_idx(3)); %peaks_idx(2));
    ticks_idx = sort([troughs_idx(1:3); peaks_idx(1:2)]);
    % ticks_idx = [find(islocalmin(theta_info.thetaLFP(theta_short_window)) | islocalmax(theta_info.thetaLFP(theta_short_window))), length(theta_short_window)];

    plot(theta_short_window, LFPFromSelectedChannels(sel_channel_idx,theta_short_window), 'b', 'DisplayName', 'raw LFP');
    hold on
    plot(theta_short_window, theta_info.thetaLFP(theta_short_window,2), '--k', 'DisplayName', 'filtered LFP');
    xline(theta_window(ticks_idx(3)), '--k', 'LineWidth', 1, 'HandleVisibility','off');
    ylabel('LFP (uV)')
    xlabel('Theta phase (degrees)')
    xticks(theta_window(ticks_idx))
    xticklabels([0 180 360 540 720])
    xlim([min(theta_short_window), max(theta_short_window)])
    set(gca,'box','off')
    legend

    % yyaxis right
    % plot(theta_short_window, thetaPhase(theta_short_window,1)*100, 'r', 'DisplayName', 'theta phase');
    % ylabel('Phase (degrees)')
    % yticks([-180 -90 0 90 180]*100)
    % yticklabels([0 90 180 270 360])

    %% E. Plot histogram of firing probabilities around SWR
    subplot(3,2,5)
    h = bar(1:28, [swr_firing{c}.first_firing_prob, swr_firing{c}.firing_prob_start, swr_firing{c}.firing_prob_end, swr_firing{c}.second_firing_prob], ...
        'hist');
    set(h, 'FaceColor', [0.5 0.5 0.5], 'EdgeColor', 'white');
    hold on
    p1 = plot(1:28, swr_firing{c}.mean_firing_prob_shuffled, '--b', 'LineWidth', 0.5); %, 'DisplayName', 'mean in non-theta/non-SWR');
    p2 = plot(1:28, swr_firing{c}.mean_firing_prob_shuffled + swr_firing{c}.std_firing_prob_shuffled, '--r', 'LineWidth', 0.5); %, 'DisplayName', 'mean +/- std in non-theta/non-SWR');
    p3 = plot(1:28, swr_firing{c}.mean_firing_prob_shuffled - swr_firing{c}.std_firing_prob_shuffled, '--r', 'LineWidth', 0.5); %, 'HandleVisibility','off');
    xticks([10.5, 14.5, 18.5]);
    xticklabels({'-1', '0', '1'});
    xlabel('Normalized time');
    ylabel('Firing probability');
    hold on
    xline(10.5, '--k', 'LineWidth', 1, 'HandleVisibility','off'); % start
    xline(14.5, '--k', 'LineWidth', 1, 'HandleVisibility','off'); % peak
    xline(18.5, '--k', 'LineWidth', 1, 'HandleVisibility','off'); % end
    % legend([p1,p2])
    set(gca,'Layer','top');
    set(gca,'box','off')

    %% F. Plot histogram of firing probabilities for theta phases
    subplot(3,2,6)
    bin_centers_theta = bin_edges_theta(1:end-1) + (bin_edges_theta(2)-bin_edges_theta(1))/2;
    bar([bin_centers_theta, bin_centers_theta+360], [theta_firing.firing_prob(c,:), theta_firing.firing_prob(c,:)], 'hist');
    xticks(min(bin_edges_theta):180:max(bin_edges_theta+360));
    xticklabels(0:180:720); % shift the labels by 180 deg
    xlim([min(bin_edges_theta) max(bin_edges_theta+360)]);
    xlabel('Theta phase (degrees)')
    ylabel('Firing probability');
    hold on
    xline(bin_edges_theta(end), '--k', 'LineWidth', 1, 'HandleVisibility','off');
    set(gca,'box','off')

    % Plot p-value
    if theta_firing.p_values(c) >= 0.001
        p_text = sprintf('p = %.3f', theta_firing.p_values(c));
    else
        p_text = sprintf('p < 0.001');
    end
    text(450, max(theta_firing.firing_prob(c,:)) + 0.1*max(theta_firing.firing_prob(c,:)), ...
        p_text, 'HorizontalAlignment', 'center', 'FontSize', 8)

    %% Extra channel and neuron info for title
    if isfield(fr_map_info,'chanMap')
        area = fr_map_info.chanMap.location(fr_map_info.chanMap.site == siteMap(clusterSites(c)));
        if ~isempty(area) % visible electrode
            textArray = ['Neuron #' num2str(c) ', Channel: ' num2str(fr_map_info.chanMap.channel(c)) ', Area: ' area{1} ', Firing rate: ' sprintf('%.2g', fr_map_info.firingRate.firing_rate(c)) ' Hz'];
        else
            textArray = ['Neuron #' num2str(c) ', Channel: ' num2str(fr_map_info.chanMap.site(c)-1) ', Firing rate: ' sprintf('%.2g', fr_map_info.firingRate.firing_rate(c)) ' Hz'];
        end
    else
        textArray = ['Neuron #' num2str(c) ', Firing rate: ' sprintf('%.2g', fr_map_info.firingRate.firing_rate(c)) ' Hz'];
    end

    if iscell(textArray)
        sgtitle(strjoin(textArray, ''), 'FontSize',10);
    else
        sgtitle(textArray, 'FontSize',10);
    end

    set(findall(gcf_manual,'-property','FontName'),'FontName','Arial');
    set(findall(gcf_manual,'-property','FontSize'),'FontSize',10);
    set(findall(gcf_manual, 'type', 'text'), 'Interpreter', 'none');
    % set(findall(gcf, 'Type', 'text'), 'Interpreter','none');
    % set(findall(gcf, 'Type', 'axes'), 'TickLabelInterpreter','none');
    % set(findall(gcf, 'Type', 'legend'), 'Interpreter','none');
    % set(findall(gcf, 'Type', 'sgtitle'), 'Interpreter','none');
    % set(findall(gcf, 'Type', 'Text'), 'Interpreter','none');

    set(gcf_manual,'defaultTextInterpreter','none');
    set(groot, 'defaultTextInterpreter', 'none');

    fig=gcf_manual;

    fig.PaperUnits = 'points';
    fig.Position = [100,100,1200,1600];
    fig.Renderer = 'Painters';
    saveas(fig, fullfile(plotpath, ['cell_' num2str(c)]), 'fig');
    % saveas(fig, fullfile(plotpath, ['cell_' num2str(c)]), 'svg');
    saveas(fig, fullfile(plotpath, ['cell_' num2str(c) '.png']), 'png');
    saveas(fig, fullfile(plotpath, ['cell_' num2str(c) '.eps']), 'epsc');
    close all
  
end
end