function [placeFieldStats, mapStats] = getPlaceFields(varargin)

% Athina Apostolelli 2024
% Adapted from bz_findPlaceFields1D_AA.m (github.com/Athina-Ap/Ippocabos)

% Parse inputs 
p=inputParser;
addParameter(p,'basepath',pwd,@isstr);
addParameter(p,'firingMaps',{},@isstruct);
addParameter(p,'geometry','2D',@isstr);  % '2D' or '1D'
addParameter(p,'FRThres',[0.1,10],@isnumeric);
addParameter(p,'infoThres',0.5,@isnumeric);
addParameter(p,'IoUThres',0.5,@isnumeric);
addParameter(p,'Peak2TroughThres',0.25,@isnumeric);
addParameter(p,'threshold',0.2,@isnumeric);
addParameter(p,'minSize',0.05,@isnumeric);
addParameter(p,'maxSize',0.50,@isnumeric);
addParameter(p,'minPeak',2,@isnumeric);
addParameter(p,'minPeak2nd',0.6,@isnumeric);
addParameter(p,'sepEdge',0.0,@isnumeric);
addParameter(p,'saveMat',true,@islogical);
addParameter(p,'forceReload',true,@islogical);

parse(p,varargin{:});
basepath = p.Results.basepath;
firingMaps = p.Results.firingMaps;
threshold = p.Results.threshold;
geometry = p.Results.geometry;
FRThres = p.Results.FRThres;
infoThres = p.Results.infoThres;
IoUThres = p.Results.IoUThres;
Peak2TroughThres = p.Results.Peak2TroughThres;

sizeMaze = size(firingMaps.rateMaps{1}{1},1);

minSize = p.Results.minSize * sizeMaze;
maxSize = p.Results.maxSize * sizeMaze;
sepEdge = p.Results.sepEdge * sizeMaze;
minPeak = p.Results.minPeak;
minPeak2nd = p.Results.minPeak2nd;
saveMat = p.Results.saveMat;
forceReload = p.Results.forceReload;

%% Load data

% Load place fields
if forceReload && ~isempty(dir([basepath filesep '*.placeFieldStats.cellinfo.mat']))
    disp('Loading placeFieldStats');
    file = dir([basepath filesep '*.placeFieldStats.cellinfo.mat']);
    load(file(1).name);
    return
end

% Load spatial information
if ~isempty(dir([basepath filesep '*.spatialInfo.cellinfo.mat']))
    disp('Loading spatialInfo');
    file = dir([basepath filesep '*.spatialInfo.cellinfo.mat'])
    load(file(1).name);
else
    disp('spatialInfo is not in this directory...');
    return
end

% Load waveforms
if ~isempty(dir([basepath filesep '*.waveforms.cellinfo.mat']))
    disp('Loading waveforms');
    file = dir([basepath filesep '*.waveforms.cellinfo.mat'])
    load(file(1).name);
else
    disp('waveforms is not in this directory...');
    return
end

%% Find place fields 
for unit = 1:length(firingMaps.rateMaps)     
    for c = 1:length(firingMaps.rateMaps{1})
        % Default values
        mapStats{unit,1}{c}.x = NaN;
        mapStats{unit,1}{c}.field = {};
        mapStats{unit,1}{c}.size = [];
        mapStats{unit,1}{c}.peak = [];
        mapStats{unit,1}{c}.mean = [];
        mapStats{unit,1}{c}.fieldX = [NaN NaN];
        mapStats{unit,1}{c}.specificity = 0;
        mapStats{unit,1}{c}.m = nan;
        mapStats{unit,1}{c}.r = nan;
        mapStats{unit,1}{c}.mode = nan;
        mapStats{unit,1}{c}.k = nan;
        mapStats{unit,1}{c}.info = spatial_info(unit); % spatial info
        mapStats{unit,1}{c}.t2p = nan; % trough to peak
        mapStats{unit,1}{c}.meanFR = nan; % mean firing rate
    
        % Determine the field as the connex area around the peak where the value or rate is > threshold*peak
        % There can be two or more fields
        z = firingMaps.rateMaps{unit}{c};
        x = 1:length(firingMaps.rateMaps{1}{1});
        
        % Maximum and mean FR along maze
        maxFR = max(max(z));
        meanFR = mean(z,'all');
    
        mapStats{unit,1}{c}.meanFR = meanFR; 
        
        % -------- Criteria ----------------------------------------------
        % If there is no firing rate, go to next unit
        if maxFR == 0
            % mapStats{unit,1}{c}.field = logical(zeros(size(z)));
            continue
        
        % If mean firing rate is less than minThres or more than maxThres, go to next unit
        elseif meanFR < FRThres(1) | meanFR > FRThres(2) 
            % mapStats{unit,1}{c}.field = logical(zeros(size(z)));
            continue           
        end

        % If spatial information content is less than infoThres, go to next unit
        if spatial_info(unit) < infoThres
            continue
        end

        % If trough to peak in waveform is less than 250usm, go to next unit
        if waveforms(unit).TroughtoPeak < 0.25 
            continue
        end
        % ----------------------------------------------------------------

        nBinsX = max([1 length(x)]);	% minimum number of bins is 1
        circX = 0; circY = 0;
        % Each time we find a field, we will remove it from the map; make a copy first
        % Try to find more fields until no remaining bin exceeds min value
        i=1;
        while true,
            % Are there any candidate (unvisited) peaks left?
            [peak,idx] = max(z(:));

            % Determine coordinates of largest candidate peak
            [y,x] = ind2sub(size(z),idx);

            % If separation from edges is less than sepEdge, go to next unit
            if strcmp(geometry,'1D')
                if (idx < sepEdge) | (idx > sizeMaze-sepEdge)
                    break;
                end
            elseif strcmp(geometry,'2D')
                if (x < sepEdge) | (x > sizeMaze-sepEdge) | (y < sepEdge) | (y > sizeMaze-sepEdge)
                    break;
                end
            end

            % If FR peak of 1st PF is less than minPeak, go to next unit
            % If FR peak of 2nd PF is less than minPeak2nd of maximum FR,
            % go to next unit
            if peak < ((i==1)*minPeak + (i>1)*maxFR*minPeak2nd)
                break;
            end
            
            % Find field (using min threshold for inclusion)
            field1 = FindFieldHelper(z,x,y,peak*threshold,circX,circY);
            if strcmp(geometry,'1D')
                size1 = sum(field1(:));
            elseif strcmp(geometry,'2D')
                size1 = [sum(max(field1,[],1)), sum(max(field1,[],2))];
            end

            % Does this field include two coalescent subfields?
            % To answer this question, we simply re-run the same field-searching procedure on the field
            % we then either keep the original field or choose the subfield if the latter is less than
            % 1/2 the size of the former
            m = peak*threshold;
            field2 = FindFieldHelper(z-m,x,y,(peak-m)*threshold,circX,circY);
            if strcmp(geometry,'1D')
                size2 = sum(field2(:));
                if size2< 1/2*size1,
                    field = field2;
                    tc = ' ';sc = '*'; % for debugging messages
                else
                    field = field1;
                    tc = '*';sc = ' '; % for debugging messages
                end
            elseif strcmp(geometry,'2D')
                size2 = [sum(max(field2,[],1)), sum(max(field2,[],2))];
                if size2(1) < 1/2*size1(1) & size2(2) < 1/2*size1(2),
                    field = field2;
                    tc = ' ';sc = '*'; % for debugging messages
                else
                    field = field1;
                    tc = '*';sc = ' '; % for debugging messages
                end
            end
            
            % If rate map between place fields doesn't go below threshold,
            % discard new place field (1D)
            good2ndPF = true; 
            if i>1 & strcmp(geometry,'1D')
                field0ini = find(diff(isnan(z))==1); if length(field0ini)>1, field0ini = field0ini(2); end
                field0end = find(diff(isnan(z))==-1); if length(field0end)>1, field0end = field0end(2); end
                field1ini = find(diff(field)==1); if isempty(field1ini), field1ini = 1; end
                field1end = find(diff(field)==-1); 
                [~,idxBetwFields] = min([abs(field1ini-field0end),abs(field0ini-field1end)]);
                if idxBetwFields == 1
                    if ~any(z(field1end:field0ini)<peak*threshold), good2ndPF = false; end
                else
                    if ~any(z(field0end:field1ini)<peak*threshold), good2ndPF = false; end
                end 

            % If overlap of fields (intersection over union) is more than
            % 50%, discard new place field (2D)
            elseif i>1 & strcmp(geometry,'2D')
                PF1 = isnan(z);
                PF2 = field;
                intersection = PF1 & PF2;
                union = PF1 | PF2;
                area_intersection = sum(intersection, 'all');
                area_union = sum(union, 'all');
                IoU = area_intersection / area_union;
                if IoU > 0.5  % fields are overlapping by more than 50% 
                    good2ndPF = false;
                end
            end

            % Keep this field if its size is sufficient
            if strcmp(geometry,'1D')
                fieldSize = sum(field(:));

                if (fieldSize > minSize) && (fieldSize < maxSize) && good2ndPF
                    mapStats{unit,1}{c}.field(:,i) = field;
                    mapStats{unit,1}{c}.size(i) = fieldSize;
                    mapStats{unit,1}{c}.peak(i) = peak;
                    mapStats{unit,1}{c}.mean(i) = mean(z(field));
                    idx = find(field & z == peak);
                    [mapStats{unit,1}{c}.y(i),mapStats{unit,1}{c}.x(i)] = ind2sub(size(z),idx(1));
                    [x,y] = FieldBoundaries(field,circX,circY);
                    [mapStats{unit,1}{c}.fieldX(i,:),mapStats{unit,1}{c}.fieldY(i,:)] = FieldBoundaries(field,circX,circY);
                end
            
            elseif strcmp(geometry,'2D')
                fieldSize = [sum(max(field,[],1)), sum(max(field,[],2))];

                if (fieldSize(1) > minSize) && (fieldSize(1) < maxSize) && (fieldSize(2) > minSize) && (fieldSize(2) < maxSize) && good2ndPF
                    mapStats{unit,1}{c}.field{i} = field;
                    mapStats{unit,1}{c}.size(i,:) = fieldSize;
                    mapStats{unit,1}{c}.peak(i) = peak;
                    mapStats{unit,1}{c}.mean(i) = mean(z(field));
                    idx = find(field & z == peak);
                    [mapStats{unit,1}{c}.y(i),mapStats{unit,1}{c}.x(i)] = ind2sub(size(z),idx(1));
                    [x,y] = FieldBoundaries(field,circX,circY);
                    [mapStats{unit,1}{c}.fieldX(i,:),mapStats{unit,1}{c}.fieldY(i,:)] = FieldBoundaries(field,circX,circY);
                end
            end

            i = i + 1;
            % Only 2 fields are allowed per cell
            if i == 3, break; end
            
            % Mark field bins as visited
            z(field) = NaN;
            if all(isnan(z)), break; end
        end
    end
end

% Save output
placeFieldStats = {};

% inherit required fields from spikes cellinfo struct
placeFieldStats.UID = firingMaps.UID;
placeFieldStats.sessionName = firingMaps.sessionName;
try
    placeFieldStats.region = firingMaps.region; 
catch
    % warning('spikes.region is missing') 
end

placeFieldStats.params.sizeMaze = sizeMaze;
placeFieldStats.params.threshold = threshold;
placeFieldStats.params.minSize = minSize;
placeFieldStats.params.maxSize = maxSize;
placeFieldStats.params.sepEdge = sepEdge;
placeFieldStats.params.minPeak = minPeak;
placeFieldStats.params.minPeak2nd = minPeak2nd;
placeFieldStats.params.saveMat = saveMat;
placeFieldStats.mapStats = mapStats;

if saveMat
   save(fullfile(basepath, [firingMaps.sessionName '.placeFieldStats.cellinfo.mat'])); 
end

end


function [x,y] = FieldBoundaries(field,circX,circY)

% Find boundaries
x = find(any(field,1));
if isempty(x),
    x = [NaN NaN];
else
    x = [x(1) x(end)];
end
y = find(any(field,2));
if isempty(y),
    y = [NaN NaN];
else
    y = [y(1) y(end)];
end

% The above works in almost all cases; it fails however for circular coordinates if the field extends
% around an edge, e.g. for angles between 350° and 30°

if circX && x(1) == 1 && x(2) == size(field,2),
    xx = find(~all(field,1));
    if ~isempty(xx),
	    x = [xx(end) xx(1)];
    end
end
if circY && y(1) == 1 && y(2) == size(field,1),
    yy = find(~all(field,2));
    if ~isempty(yy),
	    y = [yy(end) yy(1)];
    end
end
end
