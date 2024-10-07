function plotChannelCC(varargin)

% Plot channel power, and channel cross-correlograms for two time
% windows-target signal combinations. 
%
% Athina Apostolelli 2024

%% Set parameters
p=inputParser;
addParameter(p,'basepath',pwd,@isstr);
addParameter(p,'sessname','',@isstr);
addParameter(p,'channelCC',[],@isstruct);
addParameter(p,'signals',{"ripple","theta"},@iscell); % target signals ('ripples' or 'theta')
addParameter(p,'modes',{"swr","theta"},@iscell); % time windows 
addParameter(p,'fbands',{"150-250","4-10"},@iscell); % frequency bands
addParameter(p,'norm','zscore',@isstr); % power normalization ('zscore' or 'raw')

parse(p,varargin{:});
basepath = p.Results.basepath;
sessname = p.Results.sessname;
channelCC = p.Results.channelCC;
signals = p.Results.signals;
modes = p.Results.modes;
fbands = p.Results.fbands;
norm = p.Results.norm;

% Load channelCC struct 
if isempty('channelCC') & exist(fullfile(basepath, sessname, [sessname '.channelCC.mat']))
    load(fullfile(basepath, sessname, [sessname '.channelCC.mat']));
end

[~,animal] = fileparts(basepath);

%% Plot channel cross-correlation matrix and mean power
figure;
set(groot, 'defaultTextInterpreter', 'none');
savepath = fullfile(basepath, sessname, 'ChannelCC');
if ~exist(savepath)
    mkdir(savepath);
end

if channelCC.params.interpDeadChs == 1
    selected_channels = channelCC.params.phys_ch_order;
else
    selected_channels = channelCC.params.selected_chs;
end

% Tick labels
xtick_labels = cell(length(selected_channels),1);
ytick_labels = cell(length(selected_channels),1);

phys_ch_ord = channelCC.params.phys_ch_order;
siteLoc = ([0, 0; 0, 50; 0, 100; 0, 150; 0, 200; 0, 250; 0, 300; 0, 350; 0, 400; 0, 450; 0, 500; 0, 550; 0, 600; 0, 650; 0, 700; 0, 750; 0, 800; 0, 850; 0, 900; 0, 950; 0, 1000; 0, 1050; 0, 1100; 0, 1150; 0, 1200; 0, 1250; 0, 1300; 0, 1350; 0, 1400; 0, 1450; 0, 1500; 0, 1550; 0, 1600; 0, 1650; 0, 1700; 0, 1750; 0, 1800; 0, 1850; 0, 1900; 0, 1950; 0, 2000; 0, 2050; 0, 2100; 0, 2150; 0, 2200; 0, 2250; 0, 2300; 0, 2350; 0, 2400; 0, 2450; 0, 2500; 0, 2550; 0, 2600; 0, 2650; 0, 2700; 0, 2750; 0, 2800; 0, 2850; 0, 2900; 0, 2950; 0, 3000; 0, 3050; 0, 3100; 0, 3150]); % (formerly mrSiteXY) Site locations (in μm) (x values in the first column, y values in the second column)
siteIdx = find(ismember(phys_ch_ord,selected_channels));
distance = [siteLoc(siteIdx,2)];

for i = 1:length(selected_channels)
    xtick_labels{i} = num2str(selected_channels(i)-1);;
    ytick_labels{i} = distance(i);
end

% z-score power 
if strcmp(norm, "zscore")
    power1 = zscore(channelCC.(modes{1}).(signals{1}).power);
    power2 = zscore(channelCC.(modes{2}).(signals{2}).power);
elseif strcmp(norm, "raw")
    power1 = channelCC.(modes{1}).(signals{1}).power;
    power2 = channelCC.(modes{2}).(signals{2}).power;
end

% Channel power plot
subplot(subplot(1,5,1));
plot(power1, 1:length(selected_channels), 'b', 'LineWidth', 3, 'DisplayName', strcat(modes{1}, ' windows')); % Inverted plot
hold on 
plot(power2, 1:length(selected_channels), 'r', 'LineWidth', 3, 'DisplayName', strcat(modes{2}, ' windows'));
set(gca, 'YTick', 1:length(selected_channels), 'YTickLabel', ytick_labels, 'FontSize',7);
xlabel(strcat(norm, " power "), 'FontSize',10); 
ylabel('Distance from top of probe (um)', 'FontSize',10);
set(gca, 'YDir', 'reverse'); % Reverse y-axis direction
set(gca, 'Box', 'off');
ylim([1,length(selected_channels)]);
legend('FontSize',10,'Interpreter', 'none','Location','northeast')

% 1st channel CC plot
subplot(1,5,2:3);
imagesc(channelCC.(modes{1}).(signals{1}).corr);
set(gca, 'XTick', 1:length(selected_channels), 'XTickLabel', xtick_labels, 'FontSize',7);
set(gca, 'YTick', 1:length(selected_channels), 'YTickLabel', ytick_labels, 'FontSize',7);
set(gca, 'Box', 'off');
xlabel('Channel number', 'FontSize',10);
ylabel('Distance from top of probe (um)', 'FontSize',10);
title(strcat(modes{1}, " windows ", fbands{1}), 'FontSize',14)
pbaspect([1 1 1]);

% 2nd channel CC plot
subplot(1,5,4:5);
imagesc(channelCC.(modes{2}).(signals{2}).corr);
set(gca, 'XTick', 1:length(selected_channels), 'XTickLabel', xtick_labels, 'FontSize',7);
set(gca, 'YTick', 1:length(selected_channels), 'YTickLabel', ytick_labels, 'FontSize',7);
set(gca, 'Box', 'off');
xlabel('Channel number', 'FontSize',10);
ylabel('Distance from top of probe (um)', 'FontSize',10);
title(strcat(modes{2}, " windows ", fbands{2}), 'FontSize',14)
pbaspect([1 1 1]);

% Adjust positions
subplotPos1 = get(gca, 'Position'); 
subplotPos2 = get(subplot(1,5,2:3), 'Position'); 
subplotPos3 = get(subplot(1,5,4:5), 'Position'); 
subplotPos2(2) = subplotPos1(2); 
subplotPos2(4) = subplotPos1(4); 
subplotPos3(2) = subplotPos1(2); 
subplotPos3(4) = subplotPos1(4); 
set(subplot(1,5,2:3), 'Position', subplotPos2); 
set(subplot(1,5,4:5), 'Position', subplotPos3); 

% Colormap
colormap('jet');
minLimit = round(min([min(channelCC.(modes{1}).(signals{1}).corr,[],'all'), min(channelCC.(modes{2}).(signals{2}).corr, [],'all')]));
maxLimit = round(max([max(channelCC.(modes{1}).(signals{1}).corr,[],'all'), max(channelCC.(modes{2}).(signals{2}).corr, [],'all')]));
colorbar('Position',[subplotPos3(1)+subplotPos3(3)+0.02,subplotPos3(2),0.02,subplotPos1(4)], 'FontSize', 10, 'Ticks',[minLimit, 0, maxLimit], 'TickLabels',[minLimit, 0, maxLimit]);  % attach colorbar to h\
clim([minLimit,maxLimit]); 


%% Save figure
fig = gcf;
fig.PaperUnits = 'points';
fig.Renderer = 'Painters';
set(gcf, 'Position', [200,200,1500,500])

if channelCC.params.interpDeadChs == 1
    filename = strcat(animal, '_interpDead_power_ccg_', norm, '_', signals{1}, '_', signals{2});
else
    filename = strcat(animal, '_power_ccg_', norm, '_', signals{1}, '_', signals{2});
end

saveas(gcf, fullfile(savepath, filename), 'png');
saveas(gcf, fullfile(savepath, filename), 'fig');
saveas(gcf, fullfile(savepath, filename), 'svg');
close all