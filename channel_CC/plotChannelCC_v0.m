function plotChannelCC(channelCC, mode)

basepath = channelCC.params.basepath;
sessname = channelCC.params.sessname;
savepath = fullfile(channelCC.params.basepath, channelCC.params.sessname, 'ChannelCCG\final');

plotPower = channelCC.params.plotPower;
plotImpedance = channelCC.params.plotImpedance;

if ~isdir(savepath)
    mkdir(savepath);
end

conditions = fieldnames(channelCC.(mode));
for f = 1:length(conditions)
    condition = conditions{f};
    figure;
    if plotImpedance | plotPower
        subplot_pos_right = [0.55, 0.1, 0.25, 0.8];
        subplot_pos_left = [0.05, 0.1, 0.2, 0.8]; 
    
        % channel CC
        subplot(1,3,2:3);
        imagesc(channelCC.(mode).(condition).corr);
        colorbar(gca,'FontSize',10);
        hold on
        pbaspect([1 1 1]);
    
        if channelCC.params.plotMRImap
            mri_map_borders = channelCC.MRImap_borders;
            mri_border_start = find(mri_map_borders==1);
            for i = 1:length(mri_border_start)-1
                rectangle('Position',[mri_border_start(i)-0.5, mri_border_start(i)-0.5, mri_border_start(i+1)-mri_border_start(i), mri_border_start(i+1)-mri_border_start(i)]);
            end
            rectangle('Position',[mri_border_start(end)-0.5, mri_border_start(end)-0.5, length(mri_map_borders)+1-mri_border_start(end), length(mri_map_borders)+1-mri_border_start(end)]);
            set(gca,'YDir','reverse');
        end
    
        tick_labels = cell(length(channelCC.params.selected_chs),1);
        for i = 1:length(channelCC.params.selected_chs)
            if ~isempty(channelCC.sig_channels.high) & ismember(channelCC.params.selected_chs(i),channelCC.sig_channels.high)
                scatter(i,i,'*','MarkerEdgeColor','red','MarkerFaceColor','red');   
            end
            if ~isempty(channelCC.sig_channels.low) & ismember(channelCC.params.selected_chs(i),channelCC.sig_channels.low)
                scatter(i,i,'*','MarkerEdgeColor','blue','MarkerFaceColor','blue');   
            end
            tick_labels{i} = num2str(channelCC.params.selected_chs(i)-1);
        end
        set(gca, 'XTick', 1:length(channelCC.params.selected_chs), 'XTickLabel', tick_labels, 'FontSize',7);
        set(gca, 'YTick', 1:length(channelCC.params.selected_chs), 'YTickLabel', tick_labels, 'FontSize',7);
        set(gca, 'Box', 'off');
        % axis square 
        
        % channel impedance
        if plotImpedance
            plotPower = false;
            subplot(subplot(1,3,1));
            plot(channelCC.impedance, 1:length(channelCC.params.selected_chs), 'b', 'LineWidth', 1.5); % Inverted plot
            xlabel('Impedance'); % Add xlabel if needed
            set(gca, 'YDir', 'reverse'); % Reverse y-axis direction
            set(gca, 'YTick', 1:length(channelCC.params.selected_chs), 'YTickLabel', tick_labels, 'FontSize',7);
            set(gca, 'Box', 'off');
            ylim([1,length(channelCC.params.selected_chs)]);
        % channel power
        elseif plotPower
            plotImpedance = false;
            subplot(subplot(1,3,1));
            plot(channelCC.(mode).(condition).power, 1:length(channelCC.params.selected_chs), 'b', 'LineWidth', 1.5); % Inverted plot
            xlabel('Power'); % Add xlabel if needed
            set(gca, 'YDir', 'reverse'); % Reverse y-axis direction
            set(gca, 'YTick', 1:length(channelCC.params.selected_chs), 'YTickLabel', tick_labels, 'FontSize',7);
            set(gca, 'Box', 'off');
            ylim([1,length(channelCC.params.selected_chs)]);
        end
    
        % Adjust positions
        subplotPos1 = get(gca, 'Position'); 
        subplotPos2 = get(subplot(1,3,2:3), 'Position'); 
        subplotPos2(2) = subplotPos1(2); 
        subplotPos2(4) = subplotPos1(4); 
        set(subplot(1,3,2:3), 'Position', subplotPos2); 
    
        % Save figure
        set(gcf, 'Position', [200,200,1200,700])
        if ~isempty(channelCC.sig_channels.low) | ~isempty(channelCC.sig_channels.high)
            if plotImpedance
                filename = strcat('sigChs_impedance_ccg_-1-1norm_', mode, '_fband_', num2str(channelCC.(mode).(condition).fband));
            elseif plotPower
                filename = strcat('sigChs_power_ccg_-1-1norm_', mode, '_fband_', num2str(channelCC.(mode).(condition).fband));
            end
            saveas(gcf, fullfile(savepath, filename), 'png');
            saveas(gcf, fullfile(savepath, filename), 'fig');
            saveas(gcf, fullfile(savepath, filename), 'svg');
        else
            if plotImpedance
                filename = strcat('impedance_ccg_-1-1norm_', mode, '_fband_', num2str(channelCC.(mode).(condition).fband));
            elseif plotPower
                filename = strcat('power_ccg_-1-1norm_', mode, '_fband_', num2str(channelCC.(mode).(condition).fband));
            end
            saveas(gcf, fullfile(savepath, filename), 'png');
            saveas(gcf, fullfile(savepath, filename), 'fig');
            saveas(gcf, fullfile(savepath, filename), 'svg');
        end
    
    else
        imagesc(channelCC.(mode).(condition).corr);
        colorbar(gca,'FontSize',10);
        hold on
    
        if channelCC.params.plotMRImap
            mri_map_borders = channelCC.MRImap_borders;
            mri_border_start = find(mri_map_borders==1);
            for i = 1:length(mri_border_start)-1
                rectangle('Position',[mri_border_start(i)-0.5, mri_border_start(i)-0.5, mri_border_start(i+1)-mri_border_start(i), mri_border_start(i+1)-mri_border_start(i)]);
            end
            rectangle('Position',[mri_border_start(end)-0.5, mri_border_start(end)-0.5, length(mri_map_borders)+1-mri_border_start(end), length(mri_map_borders)+1-mri_border_start(end)]);
            set(gca,'YDir','reverse');
        end
    
        tick_labels = cell(length(channelCC.params.selected_chs),1);
        for i = 1:length(channelCC.params.selected_chs)
            if ~isempty(channelCC.sig_channels.high) & ismember(channelCC.params.selected_chs(i),channelCC.sig_channels.high)
                scatter(i,i,'*','MarkerEdgeColor','red','MarkerFaceColor','red');   
            end
            if ~isempty(channelCC.sig_channels.low) & ismember(channelCC.params.selected_chs(i),channelCC.sig_channels.low)
                scatter(i,i,'*','MarkerEdgeColor','blue','MarkerFaceColor','blue');   
            end
            tick_labels{i} = num2str(channelCC.params.selected_chs(i)-1);
        end
        set(gcf, 'Position', [200,200,700,700])
        set(gca, 'Box', 'off');
        set(gca, 'XTick', 1:length(channelCC.params.selected_chs), 'XTickLabel', tick_labels, 'FontSize',7);
        set(gca, 'YTick', 1:length(channelCC.params.selected_chs), 'YTickLabel', tick_labels, 'FontSize',7);
        axis square 
    
        % Save figure
        if ~isempty(channelCC.sig_channels.low) | ~isempty(channelCC.sig_channels.high)
            filename = strcat('sigChs_ccg_-1-1norm_', mode, '_fband_', num2str(channelCC.(mode).(condition).fband));
            saveas(gcf, fullfile(savepath, filename), 'png');
            saveas(gcf, fullfile(savepath, filename), 'fig');
            saveas(gcf, fullfile(savepath, filename), 'svg');
        else
            filename = strcat('ccg_-1-1norm_', mode, '_fband_', num2str(channelCC.(mode).(condition).fband));
            saveas(gcf, fullfile(savepath, filename), 'png');
            saveas(gcf, fullfile(savepath, filename), 'fig');
            saveas(gcf, fullfile(savepath, filename), 'svg');
        end
    end
end
close all
end