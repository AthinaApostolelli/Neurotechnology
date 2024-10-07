function [spatial_info] = getSpatialInformation(firingMaps,varargin)

% Spatial information content is defined in  https://www.nature.com/articles/s41467-020-14611-7 
% It is used as a place cell classification criterion. 
%
% INPUT
% firingMaps 
%
% Written by Athina Apostolelli - aapostolelli@ethz.ch

p=inputParser;
addParameter(p,'basepath',pwd,@isstr);
addParameter(p,'saveMat',true,@islogical); 

parse(p,varargin{:});
basepath = p.Results.basepath;
saveMat = p.Results.saveMat;


sizeMaze = size(firingMaps.occupancy{1}{1});
numCells = length(firingMaps.UID);

spatial_info = zeros([1,numCells]);

p_occupancy = zeros(sizeMaze);
total_occupancy = sum(firingMaps.occupancy{1}{1},'all');
for i = 1:sizeMaze(1)
    for j = 1:sizeMaze(2)
        p_occupancy(i,j) = firingMaps.occupancy{1}{1}(i,j) / total_occupancy;
    end
end

bin_firing_rate = zeros(sizeMaze);
bin_spatial_info = zeros(sizeMaze);

for unit = 1:numCells
    mean_firing_rate{unit} = mean(firingMaps.rateMaps{unit}{1}(:));
    for i = 1:sizeMaze(1)
        for j = 1:sizeMaze(2)
            bin_firing_rate(i,j) = firingMaps.rateMaps{unit}{1}(i,j);

            bin_spatial_info(i,j) = p_occupancy(i,j) * (bin_firing_rate(i,j) / mean_firing_rate{unit}) * log2(bin_firing_rate(i,j) / mean_firing_rate{unit});
        end
    end

    spatial_info(unit) = sum(bin_spatial_info,'all');
end

if saveMat 
    save(fullfile(basepath, [firingMaps.sessionName '.spatialInfo.cellinfo.mat']),'spatial_info'); 
end
end

    

