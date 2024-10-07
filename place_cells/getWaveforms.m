function [waveforms] = getWaveforms(rawWaveforms,varargin)

% INPUT
% rawWaveforms has to be in the format (waveform,time)
%
% Written by Peter Gombkoto Ph.D ETH - Neurotechnology Group - pgombkoto@ethz.ch
% Adapted by Athina Apostolelli - aapostolelli@ethz.ch

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% To run in another script:
% load('amplifier_res.mat', 'meanWfLocalRaw'); % (37.4./192); %uV/bit / gain -> convert to physical uV
% 
% wfmaxSite = meanWfLocalRaw(:,1,:);  % select channel with max amplitude 
% wfmaxSite = reshape(wfmaxSite,size(wfmaxSite,1),size(wfmaxSite,3));
% rawWaveforms = double(wfmaxSite).*(37.4./192);
% rawWaveforms = rawWaveforms';
% for c = 1:size(wfmaxSite,2)
%     waveforms(c) = getWaveforms(rawWaveforms(c,:),'basepath',basepath,'sample_rate',sample_rate);
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
p=inputParser;
addParameter(p,'basepath',pwd,@isstr);
addParameter(p,'sample_rate',20000,@isnumeric);
addParameter(p,'saveMat',true,@islogical); % TODO

parse(p,varargin{:});
basepath = p.Results.basepath;
sample_rate = p.Results.sample_rate;
saveMat = p.Results.saveMat;

% time vectors, and limitations
time_vector = (linspace(0,((size(rawWaveforms,2))-1)./sample_rate,(size(rawWaveforms,2).*2))).*1000;
peak_zero_limit=find(round(time_vector,1)==0.6,1,'first');

%% Declare memory for variables.
PeakOne=zeros(size(rawWaveforms,1),3);
PeakMinusOne=zeros(size(rawWaveforms,1),3);
PeakZero=zeros(size(rawWaveforms,1),3);
PeakMinusToPeakZero=zeros(size(rawWaveforms,1),3);
for_PCA=zeros(size(rawWaveforms,1),floor(0.0002.*(2*sample_rate)));
wave=zeros(1,size(rawWaveforms,2).*2);
PeakMinusToPeakZero=zeros(size(rawWaveforms,1),1);
PeakZero_PeakOne=zeros(size(rawWaveforms,1),1);
norm_waveform=zeros(1,size(rawWaveforms,2).*2);
stop=[];
peakA=zeros(size(rawWaveforms,1),1);
peakB=zeros(size(rawWaveforms,1),1);

%% Calculate peaks and times
% loop -> waveform by waveform calculate peaks, and implicit times
% if we are using the mean waveform as input, we only loop once
for m = 1:size(rawWaveforms,1)
    try       
        wave = interp1([1:size(rawWaveforms,2)],zscore(rawWaveforms(m,:)),[1:0.5:size(rawWaveforms,2),size(rawWaveforms,2)],'spline');
        
        % (time, amplitude, index)
        PeakZero(m,1:3)=[(time_vector(find(min(wave(1:peak_zero_limit))==wave,1,'first'))),wave((find(min(wave(1:peak_zero_limit))==wave,1,'first'))),(find(min(wave(1:peak_zero_limit))==wave,1,'first'))];
            
        Ypk=[];
        Xpk=[];
        [Ypk,Xpk,~,~,~,~,~,~,~] = findpeaks_peti(wave,time_vector,'WidthReference','halfheight');
        [~,idx]=min(abs(Xpk-PeakZero(m,1)));
        minVal=Xpk(idx);
        
        PeakMinusOne(m,1:3)=[minVal,wave(find(time_vector==minVal,1,'first')), (find(time_vector==minVal,1,'first'))];
        PeakOne(m,1:3)=[(time_vector(find(max(wave(PeakZero(m,3):end))==wave,1,'first'))),(wave(find(max(wave(PeakZero(m,3):end))==wave,1,'first'))),find(max(wave(PeakZero(m,3):end))==wave,1,'first')];
    
        PeakMinusToPeakZero(m,:)=diff([PeakMinusOne(m,1), PeakZero(m,1)]);
        
        peakA(m,:)=abs(diff([wave(PeakMinusOne(m,3)),wave(PeakZero(m,3))]));
        peakB(m,:)=abs(diff([wave(PeakZero(m,3)),wave(PeakOne(m,3))]));
        
        PeakZero_PeakOne(m,:)=diff([PeakZero(m,1),PeakOne(m,1)]);
        
        norm_waveform=wave./abs(min(wave(1:peak_zero_limit)));
        stop=PeakZero(m,3)+floor(0.0002.*(2*sample_rate));
        for_PCA(m,:)=diff(norm_waveform(PeakZero(m,3):stop));
   
    catch
        disp('Error! Peaks could not be found.')
    end
end

%% Clean up and save results 
% NOTE this only considers the case where one waveform per cluster is
% provided as input to the function. 
% cleaning up the results: if waveform (P-1) starts with 0 -> out of analysis,
% extreme values out of analysis

% if first peak is not properly detected
if PeakMinusToPeakZero < 0
    array_max = islocalmax(wave(1:PeakZero(1,3)));
    if any(array_max)
        idx_max = find(array_max==1,1,'last');
    else
        idx_max = 1;
    end
    minVal1 = time_vector(idx_max);

    PeakMinusOne(1,1:3)=[minVal1,wave(find(time_vector==minVal1,1,'first')), (find(time_vector==minVal1,1,'first'))];
    PeakOne(1,1:3)=[minVal,wave(find(time_vector==minVal,1,'first')), (find(time_vector==minVal,1,'first'))];
end

% if second peak is too far away 
if abs(PeakZero_PeakOne) > 1
    [Ypk,Xpk,~,~,~,~,~,~,~] = findpeaks_peti(wave(PeakZero(1,3):end),time_vector(PeakZero(1,3):end),'WidthReference','halfheight');
    [~,idx]=min(abs(Xpk-PeakZero(1,1)));
    minVal=Xpk(idx);
    if ~isempty(minVal)
        PeakOne(1,1:3)=[minVal,wave(find(time_vector==minVal,1,'first')), (find(time_vector==minVal,1,'first'))];
    end
end
  
waveforms.PeaktoTrough = PeakMinusToPeakZero;
waveforms.TroughtoPeak = PeakZero_PeakOne;
waveforms.AB_ratio = ((peakB-peakA)./(peakA+peakB))';
waveforms.Peaks = [PeakMinusOne(1,1) PeakZero(1,1) PeakOne(1,1)];
waveforms.Amp =[PeakMinusOne(1,2) PeakZero(1,2) PeakOne(1,2)];
waveforms.for_PCA = for_PCA(1,:);
waveforms.filtWaveform = rawWaveforms(1,:);
waveforms.Time = time_vector;
waveforms.wave = wave;

x_old = [0:1:size(rawWaveforms,2)-1]./(sample_rate);  
waveforms.Time_original=x_old;

% %% Plot results
% figure(1)
% subplot(3,3,[1 2 4 5 7 8])
% % plot(x_old.*1000,(rawWaveforms(:,:)).*6,'LineWidth',1)
% plot(time_vector, wave,'LineWidth',1)
% hold on; scatter(waveforms.Peaks(:,1),waveforms.Amp(:,1),'*');
% hold on; scatter(waveforms.Peaks(:,2),waveforms.Amp(:,2),'*');
% hold on; scatter(waveforms.Peaks(:,3),waveforms.Amp(:,3),'*');
% 
% close all

% if saveMat
%     save(fullfile(basepath, [firingMaps.sessionName 'waveforms.cellinfo.mat']),'waveforms'); 
% end

end

