function spikes = loadSpikes_JRC(varargin)
% Load clustered data from the JRClust pipeline
% DO NOT change the parameters below. 
% Buzcode compatible output. Saves output to a basename.spikes.cellinfo.mat file
%
% INPUTS
%
% See description of varargin below
%
% OUTPUT
%
% spikes:               - Matlab struct following the buzcode standard (https://github.com/buzsakilab/buzcode)
%     .sessionName      - Name of recording file
%     .UID              - Unique identifier for each neuron in a recording
%     .times            - Cell array of timestamps (seconds) for each neuron
%     .spindices        - Sorted vector of [spiketime UID], useful as input to some functions and plotting rasters
%     .maxWaveformCh    - Channel # with largest amplitude spike for each neuron (0-indexed)
%     .maxWaveformCh1   - Channel # with largest amplitude spike for each neuron (1-indexed)
%     .rawWaveform      - Average waveform on maxWaveformCh (from raw .dat)
%     .filtWaveform     - Average filtered waveform on maxWaveformCh (from raw .dat)
%     .rawWaveform_std  - Average waveform on maxWaveformCh (from raw .dat)
%     .filtWaveform_std - Average filtered waveform on maxWaveformCh (from raw .dat)
%     .peakVoltage      - Peak voltage (uV)
%     .cluID            - Cluster ID
%     .shankID          - shankID
%     .processingInfo   - Processing info
%
% DEPENDENCIES: 
%
% LoadXml.m & xmltools.m (default) or bz_getSessionInfo.m
% 
% EXAMPLE CALL
% spikes = loadSpikes_JRC('basepath',pwd); Requires xml file and dat file in basepath

% version 'NT' by Athina Apostolelli - aapostolelli@ethz.ch 
% Adapted from Peter Petersen - petersen.peter@gmail.com (last edited: 08-11-2019)


p = inputParser;
addParameter(p,'basepath',pwd,@ischar); % basepath with dat file, used to extract the waveforms from the dat file
addParameter(p,'clusteringpath','',@ischar); % clustering path to spike data
addParameter(p,'clusteringformat','jrclust',@ischar); % clustering format: [Current options: jrclust only]
addParameter(p,'basename','',@ischar); % The basename file naming convention
addParameter(p,'shanks',nan,@isnumeric); % shanks: Loading only a subset of shanks (only applicable to Klustakwik)
addParameter(p,'saveMat',true,@islogical); % Save spikes to mat file?
addParameter(p,'forceReload',true,@islogical); % Reload spikes from original format (overwrites existing mat file if saveMat==true)
addParameter(p,'getWaveforms',false,@islogical); % TODO: Get average waveforms? 
addParameter(p,'useNeurosuiteWaveforms',false,@islogical); % Use Waveform features from spk files. Alternatively it loads waveforms from dat file (Klustakwik specific)
addParameter(p,'spikes',[],@isstruct); % Load existing spikes structure to append new spike info
addParameter(p,'LSB',0.195,@isnumeric); % Least significant bit (LSB in uV) Intan = 0.195, Amplipex = 0.3815. (range/precision)
addParameter(p,'session',[],@isstruct); % A buzsaki lab db session struct
addParameter(p,'buzcode',false,@islogical); % If true, uses bz_getSessionInfo. Otherwise uses LoadXml

parse(p,varargin{:})

basepath = p.Results.basepath;
clusteringpath = p.Results.clusteringpath;
clusteringFormat = p.Results.clusteringformat;
basename = p.Results.basename;
shanks = p.Results.shanks;
forceReload = p.Results.forceReload;
saveMat = p.Results.saveMat;
getWaveforms = p.Results.getWaveforms;
spikes = p.Results.spikes;
useNeurosuiteWaveforms = p.Results.useNeurosuiteWaveforms;
LSB = p.Results.LSB;
session = p.Results.session;
buzcode = p.Results.buzcode;

% Loads parameters from a session struct
[~,basename] = fileparts(basepath);
clusteringpath = basepath;

if exist(fullfile(clusteringpath,[basename,'.spikes.cellinfo.mat'])) & forceReload
    load(fullfile(clusteringpath,[basename,'.spikes.cellinfo.mat']))
    if isfield(spikes,'ts') && (~isfield(spikes,'processinginfo') || (isfield(spikes,'processinginfo') && ~strcmp(spikes.processinginfo.version,'NT') && strcmp(spikes.processinginfo.function,'loadSpikes')))
        forceReload = false;
        disp('spikes.mat structure not up to date. Reloading spikes.')
    else
        disp('loadSpikes: Loading existing spikes file')
    end
else
    forceReload = false;
    spikes = [];
end

% Loading spikes
if ~forceReload
    % Loading session info
    if buzcode
        xml = bz_getSessionInfo(basepath, 'noPrompts', true);
        xml.SampleRate = xml.rates.wideband;
    else
        if ~exist('LoadXml.m','file') || ~exist('xmltools.m','file')
            error('''LoadXml.m'' and ''xmltools.m'' is not in your path and is required to load the xml file. If you have buzcode installed, please set ''buzcode'' to true in the input parameters.')
        else
            if contains(basepath, 'concatenated')
                if exist(fullfile(clusteringpath,'128ch_concat_data.xml'),'file')
                    xml = LoadXml(fullfile(clusteringpath,'128ch_concat_data.xml'));
                end
            else
                if exist(fullfile(clusteringpath,'amplifier.xml'),'file')
                    xml = LoadXml(fullfile(clusteringpath,'amplifier.xml'));
                end
            end
        end
    end
    switch lower(clusteringFormat)
        % Loading JRCLust 
        case 'jrclust'
            disp('loadSpikes: Loading JRClust data')
            if contains(basepath, 'concatenated')
                clusteringfile = load(fullfile(clusteringpath,'concat_data_res.mat'));
            else
                clusteringfile = load(fullfile(clusteringpath,'amplifier_res.mat'));
            end
            unit_nb = 0;
            shanks = 1:size(xml.AnatGrps,2);
            
            for shank = shanks 
                disp(['Loading shank #' num2str(shank) '/' num2str(length(shanks)) ]);
                clusters_shank = find(ismember(clusteringfile.clusterSites(:), xml.AnatGrps(shank).Channels+1));  % these cluster idx come from the channels on the specific shank
                spike_indices = find(ismember(clusteringfile.spikeClusters(:), clusters_shank));  % these spike idx correspond to the cluster idx from above  
                cluster_index = clusteringfile.spikeClusters(spike_indices);  % these are the cluster idx for each spike 
                time_stamps = clusteringfile.spikeTimes(spike_indices);
           
                if getWaveforms & useNeurosuiteWaveforms
                    fname = fullfile(clusteringpath,[basename '.spk.' num2str(shank)]);
                    f = fopen(fname,'r');
                    waveforms = LSB * double(fread(f,'int16'));
                    samples = size(waveforms,1)/size(time_stamps,1);
                    electrodes = size(xml.ElecGp{shank},2);
                    waveforms = reshape(waveforms, [electrodes,samples/electrodes,length(waveforms)/samples]);
                end

                nb_clusters = unique(cluster_index);
                nb_clusters2 = nb_clusters;
                for i = 1:length(nb_clusters2)
                    unit_nb = unit_nb +1;
                    spikes.ts{unit_nb} = time_stamps(cluster_index == nb_clusters2(i));
                    spikes.times{unit_nb} = single(spikes.ts{unit_nb})/xml.SampleRate;  
                    spikes.shankID(unit_nb) = shank;
                    spikes.UID(unit_nb) = unit_nb;
                    spikes.cluID(unit_nb) = nb_clusters2(i);
                    spikes.cluster_index(unit_nb) = nb_clusters2(i);
                    spikes.total(unit_nb) = length(spikes.ts{unit_nb});

                    if getWaveforms & useNeurosuiteWaveforms
                        spikes.filtWaveform_all{unit_nb} = mean(waveforms(:,:,cluster_index == nb_clusters2(i)),3);
                        spikes.filtWaveform_all_std{unit_nb} = permute(std(permute(waveforms(:,:,cluster_index == nb_clusters2(i)),[3,1,2])),[2,3,1]);
                        [~,index1] = max(max(spikes.filtWaveform_all{unit_nb}') - min(spikes.filtWaveform_all{unit_nb}'));
                        spikes.maxWaveformCh(unit_nb) = xml.ElecGp{shank}(index1); % index 0;
                        spikes.maxWaveformCh1(unit_nb) = xml.ElecGp{shank}(index1)+1; % index 1;
                        spikes.filtWaveform{unit_nb} = spikes.filtWaveform_all{unit_nb}(index1,:);
                        spikes.filtWaveform_std{unit_nb} = spikes.filtWaveform_all_std{unit_nb}(index1,:);
                        spikes.peakVoltage(unit_nb) = max(spikes.filtWaveform{unit_nb}) - min(spikes.filtWaveform{unit_nb});
                    end
                end
                if getWaveforms
                    spikes.processinginfo.params.WaveformsSource = 'spk files';
                end
            end
            if getWaveforms & ~useNeurosuiteWaveforms
                spikes = GetWaveformsFromDat(spikes,xml,basepath,basename,LSB,session);
            end
            end

            % Baran's animals - we select only hippocampus out of 4 areas 
            % for shank = 1:2
            %     if shank == 1
            %         clusters_shank = find(ismember(clusteringfile.clusterSites(:), xml.AnatGrps(shank).Channels));
            %     elseif shank == 2
            %         clusters_shank = find(ismember(clusteringfile.clusterSites(:), xml.AnatGrps(shank).Channels(38:end)));
            %     end
            %     spike_indices = find(ismember(clusteringfile.spikeClusters(:), clusters_shank));
            %     cluster_index = clusteringfile.spikeClusters(spike_indices);
            %     time_stamps = clusteringfile.spikeTimes(spike_indices);
            clear spike_indices cluster_index time_stamps
                    
    end
    
    spikes.sessionName = basename;
    spikes.numcells = length(spikes.UID);

    % Generate spindices matrics
    for cc = 1:spikes.numcells
        groups{cc}=spikes.UID(cc).*ones(size(spikes.times{cc}));
    end
    
    if spikes.numcells>0
        alltimes = cat(1,spikes.times{:}); groups = cat(1,groups{:});  % from cell to array
        [alltimes,sortidx] = sort(alltimes); groups = groups(sortidx); % sort both
        spikes.spindices = [alltimes groups];
    end
    
    % Attaching info about how the spikes structure was generated
    spikes.processinginfo.function = 'loadSpikes_JRC';
    spikes.processinginfo.version = 'NT';  % 3.5 from Peter Petersen's code
    spikes.processinginfo.date = now;
    spikes.processinginfo.params.forceReload = forceReload;
    spikes.processinginfo.params.shanks = shanks;
    spikes.processinginfo.params.getWaveforms = getWaveforms;
    spikes.processinginfo.params.basename = basename;
    spikes.processinginfo.params.clusteringFormat = clusteringFormat;
    spikes.processinginfo.params.clusteringpath = clusteringpath;
    spikes.processinginfo.params.basepath = basepath;
    spikes.processinginfo.params.useNeurosuiteWaveforms = useNeurosuiteWaveforms;
    try
        spikes.processinginfo.username = char(java.lang.System.getProperty('user.name'));
        spikes.processinginfo.hostname = char(java.net.InetAddress.getLocalHost.getHostName);
    catch
        disp('Failed to retrieve system info.')
    end
    
    % Saving output to a buzcode compatible spikes file.
    if saveMat
        disp('Saving spikes.')
        save(fullfile(basepath, [basename '.spikes.cellinfo.mat']),'spikes')
    end
end


% -------------------------- HELPER FUNCTIONS ----------------------------
function spikes = GetWaveformsFromDat(spikes,xml,basepath,basename,LSB,session)
% Requires a neurosuite xml structure. 
% Bad channels must be deselected in the spike groups, or skipped beforehand
timerVal = tic;
nPull = 600; % number of spikes to pull out (default: 600)
wfWin_sec = 0.004; % Larger size of waveform windows for filterning. total width in ms
wfWinKeep = 0.0008; % half width in ms
filtFreq = [500,8000];
showWaveforms = true;

badChannels = [];

% Removing channels marked as Bad in session struct
if ~isempty(session) && isfield(session.channelTags,'Bad')
    badChannels = session.channelTags.Bad.channels;
    if ~isempty(session.channelTags.Bad.spikeGroups)
        badChannels = [badChannels,session.extracellular.electrodeGroups(session.channelTags.Bad.spikeGroups)];
    end
    badChannels = unique(badChannels);
end

% Removing channels that does not exist in SpkGrps
% badChannels = [badChannels,setdiff([xml.AnatGrps.Channels],[xml.spikeGroups.Channels])+1];
badChannels = [badChannels,setdiff([xml.AnatGrps.Channels],[xml.channels])+1];

% Removing channels with skip parameter from the xml
if isfield(xml.AnatGrps,'Skip')
    channelOrder = [xml.AnatGrps.Channels]+1;
    skip = find([xml.AnatGrps.Skip]);
    badChannels = [badChannels, channelOrder(skip)];
end
goodChannels = setdiff(1:xml.nChannels,badChannels);
nGoodChannels = length(goodChannels);

[b1, a1] = butter(3, filtFreq/xml.SampleRate*2, 'bandpass');

f = waitbar(0,['Getting waveforms from dat file'],'Name',['Processing ' basename]);
if showWaveforms
    fig1 = figure('Name', ['Getting waveforms for ' basename],'NumberTitle', 'off','position',[100,100,1000,800]);
end
wfWin = round((wfWin_sec * xml.SampleRate)/2);
t1 = toc(timerVal);
s = dir(fullfile(basepath,[basename '.dat']));
duration = s.bytes/(2*xml.nChannels*xml.SampleRate);
m = memmapfile(fullfile(basepath,[basename '.dat']),'Format','int16','writable',false);
DATA = m.Data;

for ii = 1 : size(spikes.times,2)
    if ishandle(f)
        waitbar(ii/size(spikes.times,2),f,['Waveforms: ',num2str(ii),'/',num2str(size(spikes.times,2)),'. ', num2str(round(toc(timerVal)-t1)),' sec/unit, ', num2str(round(toc(timerVal)/60)) ' minutes total']);
    else
        disp('Canceling waveform extraction...')
        clear rawWaveform rawWaveform_std filtWaveform filtWaveform_std
        clear DATA
        clear m
        error('Waveform extraction canceled by user')
    end
    t1 = toc(timerVal); ;
    if isfield(spikes,'ts')
        spkTmp = spikes.ts{ii}(find(spikes.times{ii} > wfWin_sec/1.8 & spikes.times{ii} < duration-wfWin_sec/1.8));
    else
        spkTmp = round(xml.SampleRate * spikes.times{ii}(find(spikes.times{ii} > wfWin_sec/1.8 & spikes.times{ii} < duration-wfWin_sec/1.8)));
    end
    
    if length(spkTmp) > nPull
        spkTmp = spkTmp(randperm(length(spkTmp)));
        spkTmp = sort(spkTmp(1:nPull));
    end
    
    % Determines the maximum waveform channel
    startIndicies = (spkTmp(1:min(100,length(spkTmp))) - wfWin)*xml.nChannels+1;
    stopIndicies =  (spkTmp(1:min(100,length(spkTmp))) + wfWin)*xml.nChannels;
    X = cumsum(accumarray(cumsum([1;stopIndicies(:)-startIndicies(:)+1]),[startIndicies(:);0]-[0;stopIndicies(:)]-1)+1);
    %     temp1 = reshape(double(m.Data(X(1:end-1))),xml.nChannels,(wfWin*2),[]);
    wf = LSB * mean(reshape(double(DATA(X(1:end-1))),xml.nChannels,(wfWin*2),[]),3);
    wfF2 = zeros((wfWin * 2),nGoodChannels);
    for jj = 1 : nGoodChannels
        wfF2(:,jj) = filtfilt(b1, a1, wf(goodChannels(jj),:));
    end
    [~, idx] = max(max(wfF2)-min(wfF2)); % max(abs(wfF(wfWin,:)));
    spikes.maxWaveformCh1(ii) = goodChannels(idx);
    spikes.maxWaveformCh(ii) = spikes.maxWaveformCh1(ii)-1;
    
    % Assigning shankID to the unit
    for jj = 1:size(xml.AnatGrps,2)
        if any(xml.AnatGrps(jj).Channels == spikes.maxWaveformCh(ii))
            spikes.shankID(ii) = jj;
        end
    end
    
    % Pulls the waveforms from the dat
    startIndicies = (spkTmp - wfWin+1);
    stopIndicies =  (spkTmp + wfWin);
    X = cumsum(accumarray(cumsum([1;stopIndicies(:)-startIndicies(:)+1]),[startIndicies(:);0]-[0;stopIndicies(:)]-1)+1);
    X = X(1:end-1) * xml.nChannels+spikes.maxWaveformCh1(ii);
    
    wf = LSB * double(reshape(DATA(X),wfWin*2,length(spkTmp)));
    wfF = zeros((wfWin * 2),length(spkTmp));
    for jj = 1 : length(spkTmp)
        wfF(:,jj) = filtfilt(b1, a1, wf(:,jj));
    end
    
    wf2 = mean(wf,2);
    rawWaveform = detrend(wf2 - mean(wf2))';
    rawWaveform_std = std((wf-mean(wf))');
    filtWaveform = mean(wfF,2)';
    filtWaveform_std = std(wfF');
    
    window_interval = wfWin-ceil(wfWinKeep*xml.SampleRate):wfWin-1+ceil(wfWinKeep*xml.SampleRate);
    spikes.rawWaveform{ii} = rawWaveform(window_interval); % keep only +- 0.8 ms of waveform
    spikes.rawWaveform_std{ii} = rawWaveform_std(window_interval);
    spikes.filtWaveform{ii} = filtWaveform(window_interval);
    spikes.filtWaveform_std{ii} = filtWaveform_std(window_interval);
    spikes.timeWaveform{ii} = ([-ceil(wfWinKeep*xml.SampleRate)*(1/xml.SampleRate):1/xml.SampleRate:(ceil(wfWinKeep*xml.SampleRate)-1)*(1/xml.SampleRate)])*1000;
%     spikes.timeWaveform{ii} = (-wfWinKeep+1/xml.SampleRate:1/xml.SampleRate:wfWinKeep)*1000;
    spikes.peakVoltage(ii) = max(spikes.filtWaveform{ii})-min(spikes.filtWaveform{ii});
    
    if ishandle(fig1)
        figure(fig1)
        subplot(2,2,1), hold off
        plot(wfF2), hold on, plot(wfF2(:,idx),'k','linewidth',2), title('Filtered waveforms across channels'), xlabel('Samples'), ylabel('uV'),hold off
        subplot(2,2,2), hold off,
        plot(wfF), title(['Peak channel waveforms (maxWaveformCh1=',num2str(spikes.maxWaveformCh1(ii)),')']), xlabel('Samples'), ylabel('uV')
        subplot(2,2,3), hold on,
        plot(spikes.timeWaveform{ii},spikes.rawWaveform{ii}), title(['Raw waveform (',num2str(ii),'/',num2str(size(spikes.times,2)),')']), xlabel('Time (ms)'), ylabel('uV')
        xlim([-0.8,0.8])
        subplot(2,2,4), hold on,
        plot(spikes.timeWaveform{ii},spikes.filtWaveform{ii}), title('Filtered waveform'), xlabel('Time (ms)'), ylabel('uV')
        xlim([-0.8,0.8])
    end
    clear wf wfF wf2 wfF2
end
if ishandle(f)
    spikes.processinginfo.params.WaveformsSource = 'dat file';
    spikes.processinginfo.params.WaveformsFiltFreq = filtFreq;
    spikes.processinginfo.params.Waveforms_nPull = nPull;
    spikes.processinginfo.params.WaveformsWin_sec = wfWin_sec;
    spikes.processinginfo.params.WaveformsWinKeep = wfWinKeep;
    spikes.processinginfo.params.WaveformsFilterType = 'butter';
    clear rawWaveform rawWaveform_std filtWaveform filtWaveform_std
    clear DATA
    clear m
    waitbar(ii/size(spikes.times,2),f,['Waveform extraction complete ',num2str(ii),'/',num2str(size(spikes.times,2)),'.  ', num2str(round(toc(timerVal)/60)) ' minutes total']);
    disp(['Waveform extraction complete. Total duration: ' num2str(round(toc(timerVal)/60)),' minutes'])
    if ishandle(fig1)
        set(fig1,'Name',['Waveform extraction complete for ' basename])
    end
    % close(f)
end
end