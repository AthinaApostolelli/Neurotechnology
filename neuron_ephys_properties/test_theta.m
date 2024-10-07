function test_theta(num_samples, fs, LFPFromSelectedChannels, new_theta_windows, thetaLFP, thetaPhase)

Time_vector_LFP = linspace(0,num_samples./fs,num_samples);
for t = 1:length(new_theta_windows)
    % theta_window = new_theta_windows(t,1)*sample_rate : new_theta_windows(t,2)*sample_rate; % in samples
    theta_window = new_theta_windows(t,1) : new_theta_windows(t,2); % in samples

    figure()
    subplot(2,1,1)
    plot(Time_vector_LFP(theta_window),LFPFromSelectedChannels(1,theta_window),'b') % raw LFP 
    hold on; 
    plot(Time_vector_LFP(theta_window),thetaLFP(theta_window,1),'--k',"LineWidth",2) % LFP filtered for theta
    xlim([Time_vector_LFP(theta_window(1)),Time_vector_LFP(theta_window(end))]);
    ylabel('Amplitude \muV')
    title('Raw and filtered LFP from selected channel')
    subplot(2,1,2)
    plot(Time_vector_LFP(theta_window),thetaPhase(theta_window,1)) % theta phase
    xlim([Time_vector_LFP(theta_window(1)),Time_vector_LFP(theta_window(end))]);
    ylabel('Theta phase')
end
end

