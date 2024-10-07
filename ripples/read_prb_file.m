function [rawRecordings,sampleRate,bitScaling,shankMap, siteMap,siteLoc,probePad]=read_prb_file

% Specify the path to your text file
if contains(pwd, 'concatenated')
    filePath = '128ch_concat_data.prm';
else
    filePath = 'amplifier.prm';
end

% Open the file for reading
fid = fopen(filePath, 'rt');
if fid == -1
    error('Cannot open file: %s', filePath);
end

% Initialize variables to hold the extracted values
bitScaling = NaN;
probePad = [];
shankMap = [];
siteLoc = [];
siteMap = [];
rawRecordings = {};
sampleRate = NaN;

% Read the file line by line
while ~feof(fid)
    line = fgetl(fid); % Read one line from the file
    
    % Search for 'bitScaling ='
    if contains(line, 'bitScaling =')
        bitScaling = sscanf(line, 'bitScaling = %f;');
    end
    
    % Search for 'probePad ='
    if contains(line, 'probePad =')
        probePad = sscanf(line, 'probePad = [%d, %d];');
    end
    
    % Search for 'shankMap ='
    if contains(line, 'shankMap =')
        shankMapStr = extractAfter(line, 'shankMap = [');
        shankMapStr = extractBefore(shankMapStr, '];');
        shankMap = str2num(shankMapStr); % Convert string to numeric array
    end
    
    % Search for 'siteLoc ='
    if contains(line, 'siteLoc =')
        siteLocStr = extractAfter(line, 'siteLoc = [');
        siteLocStr = extractBefore(siteLocStr, '];');
        siteLocLines = strsplit(siteLocStr, ';');
        siteLoc = zeros(length(siteLocLines), 2); % Initialize siteLoc matrix
        for i = 1:length(siteLocLines)
            siteLoc(i, :) = str2num(siteLocLines{i}); % Convert each line to a row in the matrix
        end
    end

    % Search for 'siteMap ='
    if contains(line, 'siteMap =')
        siteMapStr = extractAfter(line, 'siteMap = [');
        siteMapStr = extractBefore(siteMapStr, '];');
        siteMap = str2num(siteMapStr); % Convert string to numeric array
    end

    % Search for 'rawRecordings ='
    if contains(line, 'rawRecordings =')
        rawRecordingsStr = extractAfter(line, 'rawRecordings = {');
        rawRecordingsStr = extractBefore(rawRecordingsStr, '};');
        rawRecordings = strsplit(rawRecordingsStr, ', '); % Split at comma for multiple entries if needed
        % Remove single quotes from each entry
        rawRecordings = cellfun(@(x) strrep(x, '''', ''), rawRecordings, 'UniformOutput', false);
    end

    % Search for 'sampleRate ='
    if contains(line, 'sampleRate =')
        sampleRate = sscanf(line, 'sampleRate = %d;');
    end
end

% Close the file
fclose(fid);

% Display the extracted values
% fprintf('bitScaling = %f\n', bitScaling);
% fprintf('probePad = [%d, %d]\n', probePad);
% fprintf('shankMap = [%s]\n', num2str(shankMap));
% fprintf('siteLoc = \n');
% disp(siteLoc);
% fprintf('siteMap = [%s]\n', num2str(siteMap));
% fprintf('rawRecordings = {%s}\n', strjoin(rawRecordings, ', '));
% fprintf('sampleRate = %d\n', sampleRate);