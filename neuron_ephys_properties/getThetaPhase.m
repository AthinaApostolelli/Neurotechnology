function [thetaPhase, thetaLFP] = getThetaPhase(LFPFromSelectedChannels,varargin)

% Calculates the phase of theta for each data point in the theta window 
% using the hilbert transform.
%
% INPUTS
% - sample_rate             ephys sampling rate
% - f_theta                 theta frequency band
%
% OUTPUT
% thetaPhase:    phase of theta (deg) for each data point
% thetaLFP:      raw LFP trace bandpass filtered for theta
%
% HISTORY
% Athina Apostolelli 2024
% Adapted from Peter Gombkoto 2023

%% Set input parameters
p=inputParser;
addParameter(p,'basepath',pwd,@isstr);
addParameter(p,'sample_rate',20000,@isnumeric);
addParameter(p,'f_theta',[6 10],@isnumeric);

parse(p,varargin{:});
basepath = p.Results.basepath;
sample_rate = p.Results.sample_rate;
f_theta = p.Results.f_theta;

%% Calculate theta phase
% Design a bandpass filter using 'designfilt'
bpFilt = designfilt('bandpassiir', ...
    'FilterOrder',40, ... % Adjust the order based on the desired filter sharpness and stability
    'HalfPowerFrequency1', f_theta(1), ...
    'HalfPowerFrequency2', f_theta(2), ...
    'SampleRate', sample_rate, ...
    'DesignMethod', 'butter'); % Using a Butterworth filter for a flat response in the passband

% Filter the LFP signal to isolate the theta band
thetaLFP = filtfilt(bpFilt,(LFPFromSelectedChannels'));

% Calculate the instantaneous phase of theta oscillations using the Hilbert
% transform. NOTE: The Hilbert transform shifts the data by pi/2.
thetaHilbert = hilbert(thetaLFP);
thetaPhase = rad2deg(angle(thetaHilbert));

end