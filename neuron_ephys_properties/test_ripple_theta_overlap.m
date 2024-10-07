overlap = false(size(theta_windows, 1), size(right_ripples_selected, 1));
for i = 1:size(theta_windows, 1)
    % Get the start and end timestamps of event A
    start_A = theta_windows(i, 1);
    end_A = theta_windows(i, 2);
    
    % Loop through each event of type B
    for j = 1:size(right_ripples_selected, 1)
        % Get the start and end timestamps of event B
        start_B = right_ripples_selected(j, 1);
        end_B = right_ripples_selected(j, 2);
        
        % Check for overlap between event A and event B
        if start_A <= end_B && end_A >= start_B
            overlap(i, j) = true; % Set overlap flag to true
        end
    end
end

num_overlaps = numel(find(overlap));