Time_vector_LFP = linspace(0,num_samples./fs,length(LFPFromSelectedChannels));
for t = 1:length(new_theta_windows)
    % theta_window = new_theta_windows(t,1)*sample_rate : new_theta_windows(t,2)*sample_rate; % in samples
    theta_window = new_theta_windows(t,1) : new_theta_windows(t,2); % in samples

    figure()
    subplot(4,1,1)
    plot(Time_vector_LFP(theta_window),LFPFromSelectedChannels(1,theta_window),'b')
    hold on; 
    plot(Time_vector_LFP(theta_window),thetaLFP(theta_window,1),'--k',"LineWidth",2)
    xlim([Time_vector_LFP(theta_window(1)),Time_vector_LFP(theta_window(end))]);
    ylabel('Amplitude \muV')
    title('CA1 str.pir.')
    subplot(4,1,2)
    plot(Time_vector_LFP(theta_window),thetaPhase(theta_window,1))
    xlim([Time_vector_LFP(theta_window(1)),Time_vector_LFP(theta_window(end))]);
    ylabel('Phase')
end