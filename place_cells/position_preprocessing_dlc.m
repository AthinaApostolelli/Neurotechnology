function tracking = position_preprocessing_dlc(varargin)
% Calculates the position of the animal based on DeepLabCut coordinates and 
% camera TTLs. The script assumes that two LEDs are used to track the
% animal, one on the left and one on the right hemisphere.
%
% INPUTS (promted)
%
%   csv file       - output of DeepLabCut video analysis 
%   video file     - behavioral recording (.avi)
%   info.rhd       - Intan file 
%   analogin.dat   - analog input file (with TTLs) 
%
%
% OUTPUT
%
%   position - struct with the following fields
%            .sample                    TTL on and off in recording samples
%            .left_XY                   x and y coordinates (px) of left LED
%            .right_XY                  x and y coordinates (px) of right LED
%            .times                     TTL on and off in recording timeraw spike count data
%            .amplifier_sample_rate     amplifier sample rate
%            .video_sample_rate         video sample rate
%            .angle                     head angle in rad and degrees
%            .left_XY_mm                x and y coordinates (mm) of left LED
%            .right_XY_mm               x and y coordinates (mm) of right LED
%            .filename_video            name of video file 
%            .positions                 [3 x ] matrix of times (TTL on), x and y positions (mm) 
%
% Written by Peter Gombkoto Ph.D ETH - Neurotechnology Group - pgombkoto@ethz.ch
% Adapted by Athina Apostolelli - aapostolelli@ethz.ch

p = inputParser;
addParameter(p,'basepath',pwd,@ischar); % basepath with dat file
addParameter(p,'forceReload',false,@islogical)
addParameter(p,'project','',@ischar);
addParameter(p,'mazeSize',500,@isnumeric);

parse(p,varargin{:})
basepath = p.Results.basepath;
forceReload = p.Results.forceReload;
project = p.Results.project;
mazeSize = p.Results.mazeSize;

[~,basename] = fileparts(basepath);
animal = extractBetween(basepath, 'Rat_Recording\', basename);

if exist(fullfile(basepath,[basename,'.position_dlc.mat'])) & forceReload
    load(fullfile(basepath,[basename,'.position_dlc.mat']));
    assignin('base', 'tracking', tracking);
    return
else
    forceReload = false;
    tracking = [];
end

%% Copy video and DLC csv to recording directory 
analysispath = 'C:\Users\RECORDING\Athina\place_fields';
videopath = fullfile(analysispath, project, 'videos');
 
session = extractBefore(basename, '_');
csv_file = dir(fullfile(videopath, [session '*.csv']));
if ~exist(fullfile(basepath, csv_file.name))
    disp(['Copying ' csv_file.name '...']);
    copyfile(fullfile(videopath, csv_file.name), basepath);
end

%% Import Coordinates from DLC csv

% Set up the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 7);

% Specify range and delimiter
opts.DataLines = [4, Inf];
opts.Delimiter = ",";

% Specify column names and types
% opts.VariableNames = ["frame", "RostralX", "RostralY","Rostral_likelihood", "CaudalX", "CaudalY", "Caudal_likelihood"]; % Baran's rats
opts.VariableNames = ["frame", "left_X", "left_Y", "left_likelihood", "right_X", "right_Y", "right_likelihood"]; % Eminhan's rats
opts.VariableTypes = ["double", "double", "double", "double", "double", "double", "double"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Import the data
[file,path] = uigetfile("*.csv",'Select a DeepLabCut *.csv file');
Position_Table = readtable(fullfile(path,file), opts);

% Convert to output type
tracking.sample = [];
tracking.left_XY = [Position_Table.(opts.VariableNames{2}) Position_Table.(opts.VariableNames{3}) Position_Table.(opts.VariableNames{4})];
tracking.right_XY = [Position_Table.(opts.VariableNames{5}) Position_Table.(opts.VariableNames{6}) Position_Table.(opts.VariableNames{7})];

%% Info about video file (number of frames)
% Import the data
[file,path] = uigetfile("*.avi",'Select a video file for recording');
info = mmfileinfo(fullfile(path,file));
obj = VideoReader(fullfile(path,file));

%% Detect rising edge of the TTL AI0 from Intan
[frequency_parameters, board_adc_channels] = read_Intan_RHD2000_file;
cd(path)
sample_rate = frequency_parameters.amplifier_sample_rate;
num_channels = length(board_adc_channels); % ADC input info from header file
fileinfo = dir('analogin.dat');
num_samples = fileinfo.bytes/(num_channels * 2); % uint16 = 2 bytes
fid = fopen('analogin.dat', 'r');
v = fread(fid, [num_channels, num_samples], 'uint16');
fclose(fid);
TTL_frame_video = (v-32768) * 0.0003125; % convert to volts

%% sample-frame = alignment
% onset and offset of the TTL
sdnum=4;
meanchannel=mean(TTL_frame_video); 
sdchannel=std(double(TTL_frame_video));
crossed=TTL_frame_video>meanchannel+(sdnum*sdchannel);
ShiftedData=int16((crossed-circshift(crossed,1)));

TTL_ON_OFF = zeros(length(find(ShiftedData == 1)), 2);
TTL_ON_OFF(:,1)=find(ShiftedData==1)';
TTL_ON_OFF(:,2)=find(ShiftedData==-1)';

% clear sdnum meanchannel sdchannel crossed ShiftedData
disp(['Number of TTLs for frames on analog input: ' num2str(length(TTL_ON_OFF))]);
disp(['Number of frames for the video files: ' num2str(obj.NumFrames), ' - FrameRate: ' num2str(obj.FrameRate)]);
disp(['frame difference: #TTLs - #videoFrames: ' num2str(length(TTL_ON_OFF) - obj.NumFrames)]);

% % New method
% first_TTL = find(diff(TTL_frame_video)~=0,1,'first');
% last_TTL = find(TTL_frame_video>2,1,'last')+1;
% sdnum=7;
% meanchannel=mean(TTL_frame_video(first_TTL:last_TTL)); 
% sdchannel=std(double(TTL_frame_video(first_TTL:last_TTL)));
% crossed=TTL_frame_video>meanchannel+(sdnum*sdchannel);
% ShiftedData=int16((crossed-circshift(crossed,1)));
% 
% TTL_ON_OFF = zeros(length(find(ShiftedData == 1)), 2);
% TTL_ON_OFF(:,1)=find(ShiftedData==1)';
% TTL_ON_OFF(:,2)=find(ShiftedData==-1)';
% 
% disp(['Number of TTLs for frames on analog input: ' num2str(length(TTL_ON_OFF))]);
% disp(['Number of frames for the video files: ' num2str(obj.NumFrames), ' - FrameRate: ' num2str(obj.FrameRate)]);
% disp(['frame difference: #TTLs - #videoFrames: ' num2str(length(TTL_ON_OFF) - obj.NumFrames)]);

% if this number is very big there is a problem. Usually there are more TTls
% on analog input than frames in video file. Probably the video recorder stop on PC and send a stop
% signal to camera, but this takes time to arrive and stop the camera FPGA to generate TTLs.

%% Fix outliers
differ_frame_num = length(TTL_ON_OFF) - obj.NumFrames;
tracking.sample = TTL_ON_OFF(1:end-differ_frame_num,:);
tracking.times = tracking.sample / sample_rate;
tracking.amplifier_sample_rate = sample_rate;
tracking.video_sample_rate = obj.FrameRate;

% Interpolate positions 

tracking.left_XY((tracking.left_XY(:,3)< 0.9),1)=NaN;
tracking.left_XY((tracking.left_XY(:,3)< 0.9),2)=NaN;
tracking.right_XY((tracking.right_XY(:,3)< 0.9),1)=NaN;
tracking.right_XY((tracking.right_XY(:,3)< 0.9),2)=NaN;


% for i = 1:length(tracking.left_XY(:,1))
%     if tracking.left_XY(i,3) < 0.9
%         tracking.left_XY(i,1) = NaN;
%         tracking.left_XY(i,2) = NaN;
%     end
%     if tracking.right_XY(i,3) < 0.9
%         tracking.right_XY(i,1) = NaN;
%         tracking.right_XY(i,2) = NaN;
%     end  
% end

% Velocity outliers 
xpos = mean([tracking.left_XY(:,1), tracking.right_XY(:,1)], 2);
ypos = mean([tracking.left_XY(:,2), tracking.right_XY(:,2)], 2);
t = tracking.times(:,1);  % beginning of ttl is used 

[left_x,~] = fillmissing(tracking .left_XY(:,1),'linear','SamplePoints',1:length(tracking.left_XY(:,1)));
[left_y,~] = fillmissing(tracking.left_XY(:,2),'linear','SamplePoints',1:length(tracking.left_XY(:,2)));
[right_x,~] = fillmissing(tracking.right_XY(:,1),'linear','SamplePoints',1:length(tracking.right_XY(:,1)));
[right_y,~] = fillmissing(tracking.right_XY(:,2),'linear','SamplePoints',1:length(tracking.right_XY(:,2)));

tracking.left_XY(:,1) = left_x;
tracking.left_XY(:,2) = left_y;
tracking.right_XY(:,1) = right_x;
tracking.right_XY(:,2) = right_y;


%% Fix velocity outliers 
% remove velocity outliers 


dt = diff(t);
v=(sqrt(sum(abs(diff(tracking.left_XY(:,1:2))).^2,2)))./dt; % velocity [mm/sec]

% v = zeros([length(tracking.left_XY(:,1))-1,1]);
% for i = 1:length(tracking.left_XY(:,1))-1
%     dt = t(i+1) - t(i);
%     vx = (xpos(i+1) - xpos(i)) / dt;
%     vy = (ypos(i+1) - ypos(i)) / dt;
%     v(i) = sqrt(vx^2+vy^2);
% end

tracking.left_XY(v>300,1) = NaN;
tracking.left_XY(v>300,2) = NaN;
tracking.right_XY(v>300,1) = NaN;
tracking.right_XY(v>300,2) = NaN;

[left_x,~] = fillmissing(tracking.left_XY(:,1),'linear','SamplePoints',1:length(tracking.left_XY(:,1)));
[left_y,~] = fillmissing(tracking.left_XY(:,2),'linear','SamplePoints',1:length(tracking.left_XY(:,2)));
[right_x,~] = fillmissing(tracking.right_XY(:,1),'linear','SamplePoints',1:length(tracking.right_XY(:,1)));
[right_y,~] = fillmissing(tracking.right_XY(:,2),'linear','SamplePoints',1:length(tracking.right_XY(:,2)));

tracking.left_XY(:,1) = left_x;
tracking.left_XY(:,2) = left_y;
tracking.right_XY(:,1) = right_x;
tracking.right_XY(:,2) = right_y;

%% create positions (pixels)
for num_frame=1:length(tracking.sample)
    Y1 = tracking.right_XY(num_frame,2);
    X1 = tracking.right_XY(num_frame,1);
    Y2 = tracking.left_XY(num_frame,2);
    X2 = tracking.left_XY(num_frame,1);
    theta = atan2(X2-X1,Y2-Y1);  % not very accurate with DLC
    tracking.angle(num_frame,1) = rad2deg(theta);
    tracking.angle(num_frame,2) = (theta);
end

%% Creat transformation matrix -> pixel to mm for the coordinate of the animal
% Constants - size of cage measured
monitor_W = mazeSize(1); % cage size in mm (small cage = 500) 
monitor_H = mazeSize(2);
disp(['cage size is: ' num2str(monitor_W) ' x ' num2str(monitor_H) 'mm'])

%% Select corners of the screen
total_frame = round(obj.Duration.*tracking.video_sample_rate);
obj.CurrentTime = (total_frame./2)./tracking.video_sample_rate; % go to middle of video

disp(['1.) Open video file: ' char(file)])
refImage = readFrame(obj);
disp('2.) Please select the corners of cage (base)!')

fig1=figure(1);
image(refImage);
axis image
hold on
txt = '\leftarrow 1^s^t point';
text(171,47,txt,'FontSize',14,'Color',[1, 0 ,0])

txt = '\leftarrow 2^n^d point';
text(569,42,txt,'FontSize',14,'Color',[1, 0 ,0])

txt = '\leftarrow 3^r^d point';
text(579,443,txt,'FontSize',14,'Color',[1, 0 ,0])

txt = '\leftarrow 4^t^h point';
text(184,447,txt,'FontSize',14,'Color',[1, 0 ,0])

title('draw poly line please follow the order of the corner point')

h = drawpolyline(gca,'Color','green');

%% Transformation (linear)
movingPoints = h.Position;

fixedPoints = [0,0; monitor_W,0; monitor_W,monitor_H; 0,monitor_H];
fixedPoints(:,1) = fixedPoints(:,1)+100; %moving the image in x by 100mm
fixedPoints(:,2) = fixedPoints(:,2)+50; %moving the image in y by 50mm

disp('3.) transformation matrix is DONE!')
tform = fitgeotrans(movingPoints,fixedPoints,'projective'); % calculate transformation matrix

disp('4.) transfrom the image')
Jregistered = imwarp(refImage,tform,'OutputView',imref2d(size(refImage)));

%% Plotting for inspection
disp('5.) plotting figures');
figure;
image(refImage);
xlabel('distance [pixel]')
ylabel('distance [pixel]')
title('original');
axis image;
figure;
image(Jregistered);
xlabel('distance [mm]')
ylabel('distance [mm]')
axis image;
title('cropped and transform image');

close(fig1)
pause(2)
close all

%% Transform pixel to mm
[X_transformd,Y_transformed] = transformPointsForward(tform, tracking.left_XY(:,1), tracking.left_XY(:,2));
Position_Table.leftX_mm = X_transformd;
Position_Table.leftY_mm = Y_transformed;
tracking.leftXY_mm = [X_transformd Y_transformed];

[X_transformd,Y_transformed] = transformPointsForward(tform, tracking.right_XY(:,1), tracking.right_XY(:,2));
Position_Table.rightX_mm = X_transformd;
Position_Table.rightY_mm = Y_transformed;
Position_Table.samples_TTL_ON_OFF = tracking.sample;
Position_Table.times_TTL_ON_OFF = tracking.sample / sample_rate;
tracking.rightXY_mm = [X_transformd Y_transformed];
Position_Table.angle_deg_rad = tracking.angle;

tracking.tform = tform;
tracking.filename_video = file;

% This format is needed to calculate the firing maps
tracking.position.x = mean([tracking.leftXY_mm(:,1), tracking.rightXY_mm(:,1)], 2);
tracking.position.y = mean([tracking.leftXY_mm(:,2), tracking.rightXY_mm(:,2)], 2);
tracking.position.t = tracking.times(:,1);  % beginning of ttl is used 

%% saving variables
save(fullfile(basepath,[basename,'.position_dlc.mat']),'tracking','Position_Table')

end

