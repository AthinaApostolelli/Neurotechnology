function plotPlaceFields2D(firingMaps, mapStats, varargin)

% Athina Apostolelli 2024
% Adapted from bz_findPlaceFields1D_AA/plotPlaceFields (github.com/Athina-Ap/Ippocabos)

p = inputParser;
addParameter(p,'basepath',pwd,@isstr);
addParameter(p,'savePlot',true,@islogical);
addParameter(p,'overwrite',true,@islogical);

parse(p,varargin{:});
basepath = p.Results.basepath;
savePlot = p.Results.savePlot;
overwrite = p.Results.overwrite;

plotpath = fullfile(basepath, 'PlaceFields');
if savePlot
    if overwrite & isfolder(plotpath)
        rmdir(plotpath,'s');
    end
    if ~isfolder(plotpath)
        mkdir(plotpath);
    end
end

for c = 1:length(firingMaps.rateMaps{1})
    for unit = 1:length(firingMaps.UID)
        if ~isempty(mapStats{unit}{c}.field)
            figure;
            set(gcf,'Renderer','painters')
            set(gcf,'Position',[2200 200 500 500])
            if length(mapStats{unit}{c}.field) == 1 | (length(mapStats{unit}{c}.field) > 1 & isempty(mapStats{unit}{c}.field{2}))
                h = imagesc(mapStats{unit}{c}.field{1});
                h.AlphaData = ~isnan(mapStats{unit}{c}.field{1});
            elseif length(mapStats{unit}{c}.field) > 1 & (~isempty(mapStats{unit}{c}.field{1}) & ~isempty(mapStats{unit}{c}.field{2}))
                all_fields = mapStats{unit}{c}.field{1} | mapStats{unit}{c}.field{2};
                h = imagesc(all_fields);
                h.AlphaData = ~isnan(all_fields); 
            elseif length(mapStats{unit}{c}.field) > 1 & isempty(mapStats{unit}{c}.field{1}) 
                h = imagesc(mapStats{unit}{c}.field{2});
                h.AlphaData = ~isnan(mapStats{unit}{c}.field{2});   
            end
            colorbar;
        
            if savePlot
                saveas(gcf, fullfile(plotpath, ['cell_' num2str(unit) '.png']), 'png');
            end
            close all
        end
    end
end
end