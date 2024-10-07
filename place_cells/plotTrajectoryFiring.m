function plotTrajectoryFiring(spikes, tracking, varargin)

p = inputParser;
addParameter(p,'basepath',pwd,@isstr);
addParameter(p,'mazeSize',500,@isnumeric);
addParameter(p,'savePlot',true,@islogical);
addParameter(p,'overwrite',true,@islogical);

parse(p,varargin{:});
basepath = p.Results.basepath;
savePlot = p.Results.savePlot;
overwrite = p.Results.overwrite;
mazeSize = p.Results.mazeSize;

plotpath = fullfile(basepath, 'TrajectoryMaps');
if savePlot
    if overwrite & isfolder(plotpath)
        rmdir(plotpath,'s');
    end
    if ~isfolder(plotpath)
        mkdir(plotpath);
    end
end

for pf = 1:length(spikes.UID)

    % Determine time indices when neuron spikes
    [idx] = InIntervals(spikes.times{pf}, [tracking.times(1) tracking.times(end)]);
    tsBehav = spikes.times{pf}(idx);
    for tt = 1:length(tsBehav)
        [~,closestIndex] = min(abs(tracking.times(:,1)-tsBehav(tt)));
        spikeData.posIdx{pf}(tt) = closestIndex;
    end
    spikeData.pos{pf} = [tracking.position.x(spikeData.posIdx{pf}), tracking.position.y(spikeData.posIdx{pf})];

    % Plot trajectories 
    figure
    set(gcf,'Renderer','painters')
    set(gcf,'Position',[2200 200 500 500])

    plot(tracking.position.x, tracking.position.y,'Color',[160/243 160/243 160/243],'LineWidth',1.2)
    hold on 
    scatter(tracking.position.x(spikeData.posIdx{pf}), tracking.position.y(spikeData.posIdx{pf}),8,'r','filled')
    hold off
    xlim([50 mazeSize(1)+100])
    ylim([20 mazeSize(2)+100])
    set(gca,'YDir','reverse');
    
    if savePlot
        saveas(gcf, fullfile(plotpath, ['cell_' num2str(pf) '.png']), 'png');
    end
    close all

end