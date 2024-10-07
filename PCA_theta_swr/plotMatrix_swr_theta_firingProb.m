% This script plots all the cells in one session according to their firing
% with respect to the theta phase and sharp wave ripples. 
% 
% Peter Gombkoto and Athina Apostolelli 2024

basepath = 'I:\Dropbox (Yanik Lab)\Localization Manuscript 2024\RAT DATA\rEO_06\6_240205_150311';
sessname = '6_240205_150311';
plotpath = fullfile(basepath, 'FiringCorrelations');
cd(basepath)

%% Load data
% Theta firing
if exist(fullfile(basepath, [sessname '.theta_firing_binsize20.cellinfo.mat'])) & ~exist('theta_firing')
    disp('Loading theta firing...');
    load(fullfile(basepath, [sessname '.theta_firing_binsize20.cellinfo.mat']));
end

% SWR firing
if exist(fullfile(basepath, [sessname '.swr_firing.cellinfo.mat'])) & ~exist('swr_firing')
    disp('Loading SWR firing...');
    load(fullfile(basepath, [sessname '.swr_firing.cellinfo.mat']));
end

% Channel map info 
if exist(fullfile(basepath, [sessname '.fr_mapElastic_info.cellinfo.mat'])) & ~exist('fr_map_info')
    disp('Loading firing rate and map info...');
    load(fullfile(basepath, [sessname '.fr_mapElastic_info.cellinfo.mat']));
end

cells = 1:length(swr_firing);
cells(fr_map_info.site > 64) = nan; % keep only the right hemi
cells(fr_map_info.site == 61 | fr_map_info.site == 63) = nan; 
cells(isnan(cells)) = [];

channels = fr_map_info.site(cells) - 1;

%% Transform data 
theta_data = [];
theta_data_smooth = [];
swr_data = [];
swr_shuffle_data = [];

for c = 1:length(cells)
    cell_theta_data{c} = (theta_firing.firing_prob(cells(c),:));
    f = fit([1:size(cell_theta_data{c},2)]', cell_theta_data{c}', 'smoothingspline');
    cell_theta_data_smooth{c} = f(1:size(cell_theta_data{c},2))';

    % cell_swr_data{c} = [swr_firing{cells(c)}.first_firing_prob, swr_firing{cells(c)}.firing_prob_start, swr_firing{cells(c)}.firing_prob_end, swr_firing{cells(c)}.second_firing_prob];
    % cell_swr_shuffle_data{c} = swr_firing{cells(c)}.mean_firing_prob_shuffled;
    % Keep only SWR response from start to end of event 
    cell_swr_data{c} = [swr_firing{cells(c)}.firing_prob_start, swr_firing{cells(c)}.firing_prob_end];
    cell_swr_shuffle_data{c} = [swr_firing{cells(c)}.mean_firing_prob_shuffled(11:18)];
end

theta_data = cell2mat(cell_theta_data');
theta_data_smooth = cell2mat(cell_theta_data_smooth');
% theta_data_smooth = (repmat(theta_data_smooth,[1,2])); 
swr_data = cell2mat(cell_swr_data');
swr_shuffle_data = cell2mat(cell_swr_shuffle_data');
rr_swr = swr_data./swr_shuffle_data;
rr_swr = zscore(rr_swr,0,1);

% z-scoring (first within cell and then across cells)
% zscore_theta_cell = zscore(theta_data_smooth,0,2); % dim1 = column-wise / dim2 = row-wise
zscore_theta = zscore(theta_data_smooth,0,1);

% Data used for clustering and sorting
phase = linspace(-180,180,18);
phase_new = linspace(-180,180,100);
for c = 1:size(zscore_theta,1)
    zscore_theta_interp(c,:) = interp1(phase, zscore_theta(c,:), phase_new,'spline');
    theta_data_interp(c,:) = interp1(phase, theta_data(c,:), phase_new,'spline');
end

concat_theta_swr = [zscore_theta, rr_swr];

%% PCA on cells according to theta firing
% PCA
% [coeff,score, ~, ~, explained, ~ ] = pca(zscore_theta_interp(:,:));
% PCA_best = find(cumsum(explained) < 95); %80 explained

% % KernelPCA (1)
% kernel = Kernel('type', 'cosine');
% parameter = struct('numComponents', 0.90, 'kernelFunc', kernel);
% kpca1 = KernelPCA(parameter);
% kpca1.train(zscore_theta_interp(:,:));
% results = kpca1.test(zscore_theta_interp(:,:));
% PCA_best1 = find(kpca1.cumContribution < 0.90);
% 
% % KernelPCA (2)
% kernel = Kernel('type', 'gaussian', 'gamma', 1/size(rr_swr,2));
% parameter = struct('numComponents', 0.90, 'kernelFunc', kernel);
% kpca2 = KernelPCA(parameter);1
% kpca2.train(rr_swr(:,:));
% results = kpca2.test(rr_swr(:,:));
% PCA_best2 = find(kpca2.cumContribution < 0.90);

kernel = Kernel('type', 'cosine');
parameter = struct('numComponents', 0.90, 'kernelFunc', kernel);
kpca = KernelPCA(parameter);
kpca.train(concat_theta_swr(:,:));
results = kpca.test(concat_theta_swr(:,:));
PCA_best = find(kpca.cumContribution < 0.90);

%% K-means clustering
% concat_scores = [kpca1.score(:,1:3), kpca2.score(:,1:3)];
% 
% opts = statset('Display','final');
% try
%     eva = evalclusters(concat_scores,'kmeans','CalinskiHarabasz','KList',6);
% catch
%     disp('error')
% end
% [label,C] = kmeans(concat_scores,eva.OptimalK,'Replicates',100,'Options',opts);

opts = statset('Display','final');
try
    eva = evalclusters(kpca.score(:,PCA_best),'kmeans','CalinskiHarabasz','KList',max(PCA_best));
catch
    disp('error')
end
[label,C] = kmeans(kpca.score(:,PCA_best),eva.OptimalK,'Replicates',100,'Options',opts);


%% Sort cells according to PCA clustering and theta firing
cluster_sortidx = [];
cluster_border = [0];
Cluster_kmeans_Label = [];
for i = 1:max(unique(label))
    cluster_sortidx = [cluster_sortidx; find(label==i)];
    cluster_border = [cluster_border, length(find(label==i))+cluster_border(end)];
end
cluster_sorted_theta_data = zscore_theta_interp(cluster_sortidx,:);
cluster_sorted_rrSWR_data = rr_swr(cluster_sortidx,:);
% cluster_sorted_swr_data = swr_data(cluster_sortidx,:);
% cluster_sorted_swr_shuffle_data = swr_shuffle_data(cluster_sortidx,:);
cluster_sorted_cellID = cells(cluster_sortidx);
cluster_sorted_channelID = channels(cluster_sortidx);
cluster_sorted_labels = label(cluster_sortidx);

% Creat cluster vector. darab = chunk/piece in Hungarian :) 
for darab_cluster = 1:length(cluster_border)-1
    Cluster_kmeans_Label(cluster_border(darab_cluster)+1:cluster_border(darab_cluster+1)) = darab_cluster;
end
cluster_border = cluster_border+0.5;

% Sort data according to max theta firing within the cluster
theta_data_sorted = [];
rrSWR_data_sorted = [];
% swr_data_sorted = [];
% swr_shuffle_data_sorted = [];
cellID_sorted = [];
channelID_sorted = [];

for i = 1:max(unique(label))
    temp_theta_data = cluster_sorted_theta_data(cluster_sorted_labels==i,:);
    temp_rrSWR_data = cluster_sorted_rrSWR_data(cluster_sorted_labels==i,:);
    % temp_swr_data = cluster_sorted_swr_data(cluster_sorted_labels==i,:);
    % temp_shuffle_data = cluster_sorted_swr_shuffle_data(cluster_sorted_labels==i,:);
    temp_cellID = cluster_sorted_cellID(cluster_sorted_labels==i);
    temp_channelID = cluster_sorted_channelID(cluster_sorted_labels==i);

    [~,idx] = max(temp_theta_data, [], 2);
    [~,cell_sortidx] = sort(idx);

    theta_data_sorted = [theta_data_sorted; temp_theta_data(cell_sortidx,:)];
    rrSWR_data_sorted = [rrSWR_data_sorted; temp_rrSWR_data(cell_sortidx,:)];
    % swr_data_sorted = [swr_data_sorted; temp_swr_data(cell_sortidx,:)];
    % swr_shuffle_data_sorted = [swr_shuffle_data_sorted; temp_shuffle_data(cell_sortidx,:)];
    cellID_sorted = [cellID_sorted, temp_cellID(cell_sortidx)];
    channelID_sorted = [channelID_sorted, temp_channelID(cell_sortidx)];
end


%% Plot the groupped and sorted cells
sortBySite = 0;

figure;
colormap jet

% Choose the data to plot
if sortBySite
    theta_data_toPlot = theta_data_interp;
    cellID_toPlot = cells;
    channelID_toPlot = channels;
    swr_data_toPlot = rr_swr; % risk ratio
else
    theta_data_toPlot = theta_data_sorted;
    cellID_toPlot = cellID_sorted;
    channelID_toPlot = channelID_sorted;
    swr_data_toPlot = rrSWR_data_sorted; % risk ratio
end

ch_tick_labels = cell(length(cells),1);
for i = 1:length(cells)
    if sortBySite
        area = fr_map_info.location{cellID_toPlot(cells(i))};
    else
        area = fr_map_info.location{cellID_toPlot(i)};
    end

    if iscell(area)  % visible electrode
        if strcmp(area, 'Cornu ammonis 3')
            area = 'CA3';
        elseif strcmp(area, 'Cornu ammonis 1')
            area = 'CA1';
        elseif strcmp(area, 'Laterodorsal thalamic nucleus, ventrolateral part')
            area = 'TH';
        elseif strcmp(area, 'Parietal association cortex, medial area')
            area = 'PAA';
        elseif strcmp(area, 'Dentate gyrus')
            area = 'DG';
        elseif strcmp(area, 'Clear Label')
            area = 'clear';
        end
        ch_tick_labels{i} = ['\color{blue}' num2str(channelID_toPlot(i)) ' ' area];
    
    else
        ch_tick_labels{i} = ['\color{black}' num2str(channelID_toPlot(i))];
    end
end

cell_tick_labels = cell(length(cells),1);
for i = 1:length(cells)
    if sortBySite
        p_value = theta_firing.p_values(cellID_toPlot(cells(i)));
    else
        p_value = theta_firing.p_values(cellID_toPlot(i));
    end

    if p_value < 0.001
        cell_tick_labels{i} = ['***' num2str(cellID_toPlot(i))];
    elseif p_value < 0.01
        cell_tick_labels{i} = ['**' num2str(cellID_toPlot(i))];
    elseif p_value < 0.05
        cell_tick_labels{i} = ['*' num2str(cellID_toPlot(i))];     
    else
        cell_tick_labels{i} = [num2str(cellID_toPlot(i))];
    end
end
  
% Plot the data 
subplot(1,2,1)
imagesc(theta_data_toPlot);
hold on;
line(repmat([1:size(theta_data_toPlot,2)]',1,length(cluster_border)),repmat(cluster_border,size(theta_data_sorted,2),1),'Color','black','LineWidth',2);

xlim([0.5,size(theta_data_toPlot,2)+0.5])
xticks(0.5:round(size(theta_data_toPlot,2)/4):size(theta_data_toPlot,2)+0.5)
xticklabels(0:90:360)
yticks(1:length(cells))
yticklabels(cell_tick_labels)
xlabel('Theta phase (degrees)')
ylabel('Cell')
yyaxis right 
ylim([0.5,length(cells)+0.5])
yticks(1:length(cells))
yticklabels(flip(ch_tick_labels))
colorbar;
clim([min(theta_data_toPlot(:)), max(theta_data_toPlot(:))]);

subplot(1,2,2)
imagesc(swr_data_toPlot);
hold on;
line(repmat([1:size(theta_data_toPlot,2)]',1,length(cluster_border)),repmat(cluster_border,size(theta_data_toPlot,2),1),'Color','black','LineWidth',2);
xline(10.5, '--w', 'LineWidth', 1, 'HandleVisibility','off'); % start
xline(14.5, '--w', 'LineWidth', 1, 'HandleVisibility','off'); % peak
xline(18.5, '--w', 'LineWidth', 1, 'HandleVisibility','off'); % end
xticks([10.5, 14.5, 18.5]);
xticklabels({'-1', '0', '1'});
yticks(1:length(cells))
yticklabels(cell_tick_labels)
xlabel('Normalized time')
ylabel('Cell')
yyaxis right 
ylim([0.5,length(cells)+0.5])
yticks(1:length(cells))
yticklabels(flip(ch_tick_labels))
colorbar;
clim([min([swr_data_toPlot(:)]), max([swr_data_toPlot(:)])]);

% Save figure
fig = gcf;
set(fig,'defaultTextInterpreter','none');
set(groot, 'defaultTextInterpreter', 'none');
fig.PaperUnits = 'points';
fig.Position = [100,100,1000,600];
fig.Renderer = 'Painters';
% saveas(fig, fullfile(plotpath, ['firingProb_matrices_v2']), 'fig');
% saveas(fig, fullfile(plotpath, ['firingProb_matrices_v2']), 'svg');
% saveas(fig, fullfile(plotpath, ['firingProb_matrices_v2']), 'png');

