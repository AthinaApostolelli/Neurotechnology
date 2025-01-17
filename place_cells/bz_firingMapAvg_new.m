function [firingMaps] = bz_firingMapAvg_new(positions,spikes,varargin)

% USAGE
% [firingMaps] = bz_firingMapAvg_new(positions,spikes,varargin)
% Calculates averaged firing map for a set of linear postions 
%
% INPUTS
%
%   spikes    - buzcode format .cellinfo. struct with the following fields
%               .times 
%   positions - [t x y ] or [t x] position matrix or
%               cell with several of these matrices (for different conditions)
%      or
%   behavior  - buzcode format behavior struct - NOT YET IMPLEMENTED
%   <options>      optional list of property-value pairs (see table below)
%
%    =========================================================================
%     Properties    Values
%    -------------------------------------------------------------------------
%     'smooth'			smoothing size in bins (0 = no smoothing, default = 2)
%     'nBins'			number of bins (default = 50)
%     'speedThresh'		speed threshold to compute firing rate
%     'minTime'			minimum time spent in each bin (in s, default = 0)
%     'mode'			interpolate' to interpolate missing points (< minTime),
%                   	or 'discard' to discard them (default)
%     'maxDistance'		maximal distance for interpolation (default = 5)
%     'maxGap'			z values recorded during time gaps between successive (x,y)
%                   	samples exceeding this threshold (e.g. undetects) will not
%                	    be interpolated; also, such long gaps in (x,y) sampling
%                 	    will be clipped to 'maxGap' to compute the occupancy map
%                 	    (default = 0.100 s)
%     'orderKalmanVel'	order of Kalman Velocity Filter (default 2)
%     'saveMat'   		- logical (default: false) that saves firingMaps file
%     'CellInspector'  	- logical (default: false) that creates an otuput
%                   	compatible with CellInspector
%
%
%
% OUTPUT
%
%   firingMaps - cellinfo struct with the following fields
%                .UID              Unique identifier for each neuron in a recording
%                .params           parameters for firing rate map calculation
%                .rateMaps         gaussian filtered rates
%                .countMaps        raw spike count data
%                .occupancy        position occupancy data
%
% Athina Apostolelli - Neurotechnology group - aapostolelli@ethz.ch 
% Adapted from Antonio FR, 10/2019

%% parse inputs
p=inputParser;
addParameter(p,'basepath',pwd,@isstr);
addParameter(p,'forceReload',true,@islogical);
addParameter(p,'smooth',2,@isnumeric);
addParameter(p,'speedThresh',0.1,@isnumeric);
addParameter(p,'nBins',50,@isnumeric);
addParameter(p,'maxGap',0.1,@isnumeric);
addParameter(p,'minTime',0,@isnumeric);
addParameter(p,'saveMat',false,@islogical);
addParameter(p,'CellInspector',false,@islogical);
addParameter(p,'mode','discard',@isstr);
addParameter(p,'maxDistance',5,@isnumeric);
addParameter(p,'orderKalmanVel',2,@isnumeric);

parse(p,varargin{:});
basepath = p.Results.basepath;
forceReload = p.Results.forceReload;
smooth = p.Results.smooth;
speedThresh = p.Results.speedThresh;
nBins = p.Results.nBins;
maxGap = p.Results.maxGap;
minTime = p.Results.minTime;
saveMat = p.Results.saveMat;
CellInspector = p.Results.CellInspector;
mode = p.Results.mode;
maxDistance = p.Results.maxDistance;
order = p.Results.orderKalmanVel;

% number of conditions
  if iscell(positions)
     conditions = length(positions); 
  else
     conditions = 1;
     temp{1} = positions;
     positions = temp;
  end
  %%% TODO: conditions label
  
%% Load firing maps
if forceReload && ~isempty(dir([basepath filesep '*.firingMapsAvg.cellinfo.mat'])) 
    disp('Loading firingMaps');
    file = dir([basepath filesep '*.firingMapsAvg.cellinfo.mat']);
    load(file(1).name);
    return
end
  
%% Calculate
% Erase positions below speed threshold
for iCond = 1:size(positions,2)
    % Compute speed
    post = positions{iCond}(:,1);

    if size(positions{iCond},2)==2  % 1D
        posx = positions{iCond}(:,2);
        [~,~,~,vx,vy,~,~] = KalmanVel(posx,posx*0,post,order);
    elseif size(positions{iCond},2)==3  % 2D
        posx = positions{iCond}(:,2);
        posy = positions{iCond}(:,3);
        [~,~,~,vx,vy,~,~] = KalmanVel(posx,posy,post,order);
    else
        warning('This is not a linear nor a 2D space!');
    end
    % Absolute speed
    v = sqrt(vx.^2+vy.^2);
    
    % Compute timestamps where speed is under threshold
    positions{iCond}(v<speedThresh,:) = [];
end

% Get firing rate maps
for unit = 1:length(spikes.times)
    for c = 1:conditions
        map{unit}{c} = Map_AA(positions{c},spikes.times{unit},'smooth',smooth,'minTime',minTime,...
            'nBins',nBins,'maxGap',maxGap,'mode',mode,'maxDistance',maxDistance);
    end
end
%%% TODO: pass rest of inputs to Map

%% Restructure into cell info data type

% Inherit required fields from spikes cellinfo struct
firingMaps.UID = spikes.UID;
firingMaps.sessionName = spikes.sessionName;
firingMaps.params.smooth = smooth;
firingMaps.params.minTime = minTime;
firingMaps.params.nBins = nBins;
firingMaps.params.maxGap = maxGap;
firingMaps.params.mode = mode;
firingMaps.params.maxDistance = maxDistance;


for unit = 1:length(spikes.times)
    for c = 1:conditions
    firingMaps.rateMaps{unit,1}{c} = map{unit}{c}.z;
    firingMaps.countMaps{unit,1}{c} = map{unit}{c}.count;
    firingMaps.occupancy{unit,1}{c} = map{unit}{c}.time;
    %Get the x bins back in units of meters...
    % firingMaps.xbins{unit,1}{c} = map{unit}{c}.x .* ...
    %     (max(positions{c}(:,2))-min(positions{c}(:,2))) + min(positions{c}(:,2));
    end
end

if saveMat
   save(fullfile(basepath, [firingMaps.sessionName '.firingMapsAvg.cellinfo.mat']),'firingMaps'); 
end

end
