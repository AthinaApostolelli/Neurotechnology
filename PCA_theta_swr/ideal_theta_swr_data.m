num_samples_per_cluster = 25;
num_total_samples = 8*num_samples_per_cluster;

%% Theta data 
frequency = 10; % Hz
time_theta =  [0:360*frequency/18:360*frequency-360*frequency/18];
phases = [0:90:270];  

theta_data = [];
for t = 1:length(phases)
    jitterPhase=30;
    theta_jitter = -jitterPhase + (jitterPhase+jitterPhase).*rand(num_samples_per_cluster*2,1);

    sine_wave = 0.2*sind(1/frequency*time_theta + phases(t) + theta_jitter) + 0.2;
    % phi = (1/frequency) * time_theta + phases(t) + theta_jitter;
    % phi = mod(phi, 360);
    theta_data = [theta_data; sine_wave];
    % figure
    % hold on
    % plot(time_theta, sine_wave(:,:))   
end

%% SWR data
time_swr = [-1:2/8:1-2/8];
swr_mean = -0.25;
swr_std = 0.25;
swr_responses = [0,1]; % 1 = excitation, 0 = inhibition

exc_inh_idx = randi([0,1],[num_total_samples,1]);

swr_data = [];
swr_jitter = -0.1 + (0.1+0.1).*rand(num_total_samples,1);
for c = 1:num_total_samples
    if exc_inh_idx(c) == 1
        swr_tuning(c,:) = 0.3 * exp(-(time_swr - swr_mean + swr_jitter(c)).^2 ./ (2 * swr_std^2)) + 0.05;
    else 
        swr_tuning(c,:) = - 0.3 * exp(-(time_swr - swr_mean + swr_jitter(c)).^2 ./ (2 * swr_std^2)) + 0.05 + 0.3; % Change this 
    end
end
swr_data = swr_tuning;    

%% Artificial cell data
artificial_cells = 1:num_total_samples;

all_data = [theta_data, swr_data]; % 26 bins 

%% PCA
option = 2;
switch option
    case 1
        % Option (1): PCA
        [coeff,score, ~, ~, explained, ~ ] = pca(all_data(:,:));
        PCA_best = find(cumsum(explained) < 95); % variance explained
        
        figure(1)
        scatter3(score(:,1),score(:,2),score(:,3))
    case 2
        % Option (2): KPCA 2 kernels
        % KernelPCA (1)
        kernel = Kernel('type', 'cosine');
        parameter = struct('numComponents', 0.90, 'kernelFunc', kernel);
        kpca1 = KernelPCA(parameter);
        kpca1.train(all_data(:,:));
        results = kpca1.test(all_data(:,:));
        PCA_best1 = find(kpca1.cumContribution < 0.90);
        
        figure(2)
        scatter3(kpca1.score(:,1),kpca1.score(:,2),kpca1.score(:,3))
        title('cosine')

        % KernelPCA (2)
        kernel = Kernel('type', 'gaussian', 'gamma', 1/size(all_data,2));
        parameter = struct('numComponents', 0.90, 'kernelFunc', kernel);
        kpca2 = KernelPCA(parameter);1
        kpca2.train(all_data(:,:));
        results = kpca2.test(all_data(:,:));
        PCA_best2 = find(kpca2.cumContribution < 0.90);
        
        figure(3)
        scatter3(kpca2.score(:,1),kpca2.score(:,2),kpca2.score(:,3))
        title(['gaussian, gamma = 1/' num2str(size(all_data,2))])

        % score = [kpca1.score(:,1:4), kpca2.score(:,1:2)];
    
    case 3
        % Option (3): KPCA sum 2 kernels 
        kernel = Kernel('type', 'sigmoid');
        parameter = struct('numComponents', 0.90, 'kernelFunc', kernel);
        kpca = KernelPCA(parameter);
        kpca.train(all_data(:,:));
        results = kpca.test(all_data(:,:));
        PCA_best = find(kpca.cumContribution < 0.99);
        score = kpca.score;

        figure(4)
        scatter3(score(:,1),score(:,2),score(:,3))
end

%% k-means clustering
opts = statset('Display','final');
try
    eva = evalclusters(score,'kmeans','CalinskiHarabasz','KList',3);
catch
    disp('error')
end
[label,C] = kmeans(score,8,'Replicates',100,'Options',opts);

%% Sort cells according to PCA clustering and theta firing
cluster_sortidx = [];
cluster_border = [0];
Cluster_kmeans_Label = [];
for i = 1:max(unique(label))
    cluster_sortidx = [cluster_sortidx; find(label==i)];
    cluster_border = [cluster_border, length(find(label==i))+cluster_border(end)];
end
cluster_sorted_data = all_data(cluster_sortidx,:);
cluster_sorted_theta_data = theta_data(cluster_sortidx,:);
cluster_sorted_swr_data = swr_data(cluster_sortidx,:);

% Creat cluster vector. darab = chunk/piece in Hungarian :) 
for darab_cluster = 1:length(cluster_border)-1
    Cluster_kmeans_Label(cluster_border(darab_cluster)+1:cluster_border(darab_cluster+1)) = darab_cluster;
end
cluster_border = cluster_border+0.5;

% Sort data according to max theta firing within the cluster
% data_sorted = [];
% theta_data_sorted = [];
% swr_data_sorted = [];
% 
% for i = 1:max(unique(label))
%     temp_data = cluster_sorted_data(cluster_sorted_labels==i,:);
%     temp_theta_data = cluster_sorted_theta_data(cluster_sorted_labels==i,:);
%     temp_swr_data = cluster_sorted_swr_data(cluster_sorted_labels==i,:);
% 
%     [~,idx] = max(temp_theta_data, [], 2);
%     [~,cell_sortidx] = sort(idx);
% 
%     theta_data_sorted = [theta_data_sorted; temp_theta_data(cell_sortidx,:)];
%     rrSWR_data_sorted = [rrSWR_data_sorted; temp_rrSWR_data(cell_sortidx,:)];
%     % swr_data_sorted = [swr_data_sorted; temp_swr_data(cell_sortidx,:)];
%     % swr_shuffle_data_sorted = [swr_shuffle_data_sorted; temp_shuffle_data(cell_sortidx,:)];
%     cellID_sorted = [cellID_sorted, temp_cellID(cell_sortidx)];
%     channelID_sorted = [channelID_sorted, temp_channelID(cell_sortidx)];
% end


%% Plot the groupped and sorted cells
sortBySite = 0;

figure;
colormap jet

% Plot the data 
subplot(1,2,1)
imagesc(cluster_sorted_theta_data);
hold on;
line(repmat([1:size(cluster_sorted_theta_data,2)]',1,length(cluster_border)),repmat(cluster_border,size(cluster_sorted_theta_data,2),1),'Color','black','LineWidth',2);

xlim([0.5,size(cluster_sorted_theta_data,2)+0.5])
xticks(0.5:round(size(cluster_sorted_theta_data,2)/4):size(cluster_sorted_theta_data,2)+0.5)
xticklabels(0:90:360)
xlabel('Theta phase (degrees)')
ylabel('Cell')
colorbar;
clim([min(cluster_sorted_theta_data(:)), max(cluster_sorted_theta_data(:))]);

subplot(1,2,2)
imagesc(cluster_sorted_swr_data);
hold on;
line(repmat([1:size(cluster_sorted_theta_data,2)]',1,length(cluster_border)),repmat(cluster_border,size(cluster_sorted_theta_data,2),1),'Color','black','LineWidth',2);
xline(10.5, '--w', 'LineWidth', 1, 'HandleVisibility','off'); % start
xline(14.5, '--w', 'LineWidth', 1, 'HandleVisibility','off'); % peak
xline(18.5, '--w', 'LineWidth', 1, 'HandleVisibility','off'); % end
% xticks([10.5, 14.5, 18.5]);
% xticklabels({'-1', '0', '1'});
xlabel('Normalized time')
ylabel('Cell')
colorbar;
clim([min([cluster_sorted_swr_data(:)]), max([cluster_sorted_swr_data(:)])]);