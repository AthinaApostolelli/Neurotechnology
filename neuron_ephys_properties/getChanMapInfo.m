function [chanMap] = getChanMapInfo(varargin)

%% Set parameters
p = inputParser;    
addParameter(p,'basepath',pwd,@isfolder);
addParameter(p,'mapFile','',@ischar);
addParameter(p,'clusterSites',[],@isnumeric);
addParameter(p,'siteMap',[],@isnumeric);

parse(p,varargin{:});
basepath = p.Results.basepath;
mapFile = p.Results.mapFile;
clusterSites = p.Results.clusterSites;
siteMap = p.Results.siteMap;

if isempty(mapFile)
    error('Pelase provide a valid file path for the MRI map.');
end

cd(basepath);

%% Load data 
% Spiking and electrode data
if isempty(siteMap)
    [rawRecordings,sampleRate,bitScaling,shankMap,siteMap,siteLoc,probePad] = read_prb_file; % read paramater about the electrode, and raw data file.
end

if isempty(clusterSites)
    if contains(basepath, 'concatenated')
        load(fullfile(basepath, 'concat_data_res.mat'), 'clusterSites');
    else
        load(fullfile(basepath, 'amplifier_res.mat'), 'clusterSites');
    end
end

% Spikes
spikes = loadSpikes_JRC('basepath',basepath,'forceReload',false);
num_neurons = length(spikes.UID);

%% Load the MRI map
fileID = fopen(mapFile, 'r');
data = textscan(fileID, '%s', 'Delimiter', '\n');
fclose(fileID);

lines = data{1};
num_lines = numel(lines);
site = zeros(num_lines, 1);
channel = zeros(num_lines, 1);
location = cell(num_lines, 1);
segment_number = zeros(num_lines, 1);

% Parse each line
for i = 1:num_lines
    line = lines{i};
    parts = regexp(line, 'CH:(\d+) in (.+) Segment: (\d+\.?\d*)', 'tokens');
    if ~isempty(parts)
        site(i) = str2double(parts{1}{1})+1; % original site number 0-idx, here we switch to 1-idx to match JRC
        channel(i) = str2double(parts{1}{1}); % original channel
        location{i} = parts{1}{2};
        segment_number(i) = str2double(parts{1}{3});
    end
end

for c = 1:num_neurons
    area = location(site == siteMap(clusterSites(c)));
    if ~isempty(area) % visible electrode
        chanMap.site(c) = siteMap(clusterSites(c));
        chanMap.channel(c) = channel(site == siteMap(clusterSites(c)));
        chanMap.location{c} = area;
        chanMap.segment(c) = segment_number(site == siteMap(clusterSites(c)));
    else
        chanMap.site(c) = siteMap(clusterSites(c));
        chanMap.channel(c) = NaN;
        chanMap.location{c} = NaN;
        chanMap.segment(c) = NaN;
    end
end

end