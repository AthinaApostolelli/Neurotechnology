function plotFiringMaps(firingMaps, varargin)

p = inputParser;
addParameter(p,'basepath',pwd,@isstr);
addParameter(p,'savePlot',true,@islogical);
addParameter(p,'overwrite',true,@islogical);

parse(p,varargin{:});
basepath = p.Results.basepath;
savePlot = p.Results.savePlot;
overwrite = p.Results.overwrite;

plotpath = fullfile(basepath, 'FiringMaps');
if savePlot
    if overwrite & isfolder(plotpath)
        rmdir(plotpath,'s');
    end
    if ~isfolder(plotpath)
        mkdir(plotpath);
    end
end

for pf = 1:length(firingMaps.UID)
    figure
    set(gcf,'Renderer','painters')
    set(gcf,'Position',[2200 200 500 500])

    h = imagesc(firingMaps.rateMaps{pf}{1});
    set(h, 'AlphaData', ~isnan(firingMaps.rateMaps{pf}{1}))
    set(gca,'YDir','reverse');
    colorbar;

    if savePlot
        saveas(gcf, fullfile(plotpath, ['cell_' num2str(pf) '.png']), 'png');
    end
    close all
end

end