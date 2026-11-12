%clear all
close all;  clear all;  clc;

addpath(genpath('indexes'));
addpath(genpath(cd));


% path_1 = 'D:\Bai5\DATFuse - Copy\MyDatasets\DatasetBMP\SPECT-MRI\test\MRI';
% path_2 = 'D:\Bai5\DATFuse - Copy\MyDatasets\DatasetBMP\SPECT-MRI\test\SPECT'; 
% path_3 = 'D:\Bai5\DATFuse - Copy\HaResults\gray\SPECT-MRI'; 
% 
% % Define output member and create paths
% member = "MRI_SPECT";


path_1 = 'C:\Users\Admin\MIF\MIF\datasets\Dataset_coHa\CT-MRI\test\CT';
path_2 = 'C:\Users\Admin\MIF\MIF\datasets\Dataset_coHa\CT-MRI\test\MRI'; 
path_3 = 'C:\Users\Admin\MIF\LRFNet\results_coHa\results_gray\CT-MRI'; 


% path_1 = 'C:\Users\Admin\MIF\DatasetBMP_update\CT-MRI\test\CT';
% path_2 = 'C:\Users\Admin\MIF\DatasetBMP_update\CT-MRI\test\MRI'; 
% path_3 = 'C:\Users\Admin\MIF\MIF\post_processed_results\dual_lossDATFuse\CT-MRI\results_gray'; 

% Define output member and create paths
member = "CT-MRI";


outputFolder = fullfile('C:\Users\Admin\MIF\LRFNet\results_coHa', member);
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% Define text file for results
filename_full = append(member, '_full.txt');
path_full = fullfile(outputFolder, filename_full);
fileID_slice = fopen(path_full, 'w');

% Define headers with 29 columns to match excelData's structure
headers = {'Num', 'Name', 'MLI', 'SD', 'Entropy', 'AG', 'Qabf', 'VIFF', 'FMI', 'MI', ...
           'Xydeas', 'Qp', 'Q', 'Qw', 'Qe', 'NIQE', 'SSEQ', 'FSIM', 'SSIM', 'TMQI', ...
           'PSNR', 'Chen', 'CB', 'metricWang', 'NCIE', 'metricZhao', 'x1', 'x2', 'Time'};

% Write headers with fixed-width formatting
headerFormat = '%-5s %-15s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s\n';
fprintf(fileID_slice, headerFormat, headers{:});

% Initialize arrays for results
name_f = [];
v1_f = [];
y_f = [];
time_f = [];

% Read image files
original_files_1 = [dir(fullfile(path_1, '*.bmp')); dir(fullfile(path_1, '*.png')); dir(fullfile(path_1, '*.jpg')); dir(fullfile(path_1, '*.jpeg'))];
original_files_2 = [dir(fullfile(path_2, '*.bmp')); dir(fullfile(path_2, '*.png')); dir(fullfile(path_2, '*.jpg')); dir(fullfile(path_2, '*.jpeg'))];
original_files_3 = [dir(fullfile(path_3, '*.bmp')); dir(fullfile(path_3, '*.png')); dir(fullfile(path_3, '*.jpg')); dir(fullfile(path_3, '*.jpeg'))];

[~, idx1] = sort({original_files_1.name});
[~, idx2] = sort({original_files_2.name});
[~, idx3] = sort({original_files_3.name});
original_files_1 = original_files_1(idx1);
original_files_2 = original_files_2(idx2);
original_files_3 = original_files_3(idx3);

numImages = min([length(original_files_1), length(original_files_2), length(original_files_3)]);

if numImages == 0
    error('No supported image files found in one or more evaluation folders.');
end

if length(original_files_1) ~= length(original_files_2) || length(original_files_1) ~= length(original_files_3)
    warning('Image count mismatch across folders. Evaluating the first %d matched entries only.', numImages);
end

for k = 1:numImages
        disp(['Dang xu y anh thu: ', num2str(k)]);
        filename1 = fullfile(path_1, original_files_1(k).name);
        filename2 = fullfile(path_2, original_files_2(k).name);
        filename3 = fullfile(path_3, original_files_3(k).name);
        [~, baseFileNameNoExt1, ~] = fileparts(filename1);
        [~, baseFileNameNoExt2, ~] = fileparts(filename2);

        % Only proceed if filenames match
        if strcmp(baseFileNameNoExt1, baseFileNameNoExt2)
            
            A = im2double(imread(filename1));
            img2 = im2double(imread(filename2));
            
            imgf = im2double(imread(filename3));

            if size(A, 3) > 1
                A = rgb2gray(A);
            end

            [hei, wid] = size(A);
            
            
            % Convert RGB to YUV
            B_YUV = ConvertRGBtoYUV(img2);
            BB = B_YUV(:, :, 1);

            % Perform Fusion and Time Calculation
            tic

            time = toc;

            F = imgf;
            % Evaluate Metrics
            v = DanhGiaAnh(F);
            v1 = zeros(1,4);
            v1(1) = v(1);
            v1(2) = v(2);
            v1(3) = v(3);
            v1(4) = v(4);
            
            im1 = im2uint8(A);
            im2 = im2uint8(BB);
            imf = im2uint8(F);

            % Metrics calculations for y_f
            y = zeros(1, 22);
            y(1) = Qabf(im1, im2, imf);
            y(2) = VIFF(im1, im2, imf);
            y(3) = fmi(im1, im2, imf, 'edge', 3);
            y(4) = mutual_information(im1, im2, imf, 256);
            y(5) = metricXydeas(im1, im2, imf);
            y(6) = Qp_ABF(im1, im2, imf);
            y(7) = metricPeilla(im1, im2, imf, 1);
            y(8) = metricPeilla(im1, im2, imf, 2);
            y(9) = metricPeilla(im1, im2, imf, 3);
            y(10) = niqe(imf);
            y(11) = SSEQ(cat(3, im1, im2, imf));
            y(12) = FSIM(im1, imf);
            y(13) = MS_SSIM(im1, im2, imf);
            y(14) = TMQI(cat(3,im1, im1, im1), cat(3,imf, imf, imf));
            y(15) = psnr(imf,im1);
            y(16) = metricChen(im1, im2, imf);
            y(17) = metricChenBlum(im1, im2, imf);
            y(18) = metricWang(im1, im2, imf);
            y(19) = Q_NCIE(im1,im2,imf);
            y(20) = -1 % metricZhao(imresize(im1,2), imresize(im2,2), imresize(imf,2)); %Qp


            %% YUV to RGB
          

            % Save data in arrays for Excel
            name_f = [name_f; baseFileNameNoExt1];
            v1_f = [v1_f; v1];
            y_f = [y_f; y];
            time_f = [time_f; time];

            % Write to text file with aligned formatting
            rowFormat = '%-5d %-15s %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f\n';
            fprintf(fileID_slice, rowFormat, k, baseFileNameNoExt1, v1, y, time);
        end
end

% Ensure consistent dimensions for Excel data
numEntries = min([length(v1_f), size(y_f, 1), length(time_f)]);
v1_f = v1_f(1:numEntries, :);  % Ensure v1_f is of size [numEntries, 4]
y_f = y_f(1:numEntries, :);
time_f = time_f(1:numEntries);

% Calculate max, min, mean, std for numeric columns, including time
v1_max = max(v1_f, [], 1);
y_max = max(y_f, [], 1);
time_max = max(time_f);

v1_min = min(v1_f, [], 1);
y_min = min(y_f, [], 1);
time_min = min(time_f);

v1_mean = mean(v1_f, 1);
y_mean = mean(y_f, 1);
time_mean = mean(time_f);

v1_std = std(v1_f, 1);
y_std = std(y_f, 1);
time_std = std(time_f);

% Prepare each component for concatenation in excelData
rowIndices = num2cell((1:numEntries)');                   % [numEntries x 1]
%baseNames1 = repmat({baseFileNameNoExt1}, numEntries, 1);  % [numEntries x 1]
baseNames = cellstr(name_f);
v1_f_cells = num2cell(v1_f);                              % [numEntries x 4]
y_f_cells = num2cell(y_f);                                % [numEntries x 22]
time_f_cells = num2cell(time_f);                          % [numEntries x 1]

% Concatenate each component to form excelData with 29 columns
%excelData = [rowIndices, baseNames, v1_f_cells, y_f_cells, time_f_cells];
excelData = [rowIndices, baseNames, v1_f_cells, y_f_cells, time_f_cells];

% Append max, min, mean, and std to the excelData array with blanks for non-numeric columns
excelData = [excelData; ...
    {[], 'Max', v1_max(1), v1_max(2), v1_max(3), v1_max(4), y_max(1), y_max(2), y_max(3), y_max(4), y_max(5), y_max(6), y_max(7), y_max(8), y_max(9), y_max(10), y_max(11), y_max(12), y_max(13), y_max(14), y_max(15), y_max(16), y_max(17), y_max(18), y_max(19), y_max(20), y_max(21), y_max(22), time_max}; ...
    {[], 'Min', v1_min(1), v1_min(2), v1_min(3), v1_min(4), y_min(1), y_min(2), y_min(3), y_min(4), y_min(5), y_min(6), y_min(7), y_min(8), y_min(9), y_min(10), y_min(11), y_min(12), y_min(13), y_min(14), y_min(15), y_min(16), y_min(17), y_min(18), y_min(19), y_min(20), y_min(21), y_min(22), time_min}; ...
    {[], 'Mean', v1_mean(1), v1_mean(2), v1_mean(3), v1_mean(4), y_mean(1), y_mean(2), y_mean(3), y_mean(4), y_mean(5), y_mean(6), y_mean(7), y_mean(8), y_mean(9), y_mean(10), y_mean(11), y_mean(12), y_mean(13), y_mean(14), y_mean(15), y_mean(16), y_mean(17), y_mean(18), y_mean(19), y_mean(20), y_mean(21), y_mean(22), time_mean}; ...
    {[], 'Std', v1_std(1), v1_std(2), v1_std(3), v1_std(4), y_std(1), y_std(2), y_std(3), y_std(4), y_std(5), y_std(6), y_std(7), y_std(8), y_std(9), y_std(10), y_std(11), y_std(12), y_std(13), y_std(14), y_std(15), y_std(16), y_std(17), y_std(18), y_std(19), y_std(20), y_std(21), y_std(22), time_std}];

% Write data and summary statistics to Excel
% Lấy tên thư mục hiện tại
currentFolder = pwd;
[~, currentFolderName] = fileparts(currentFolder);

% Tạo đường dẫn file đầu ra với tên thư mục hiện tại trước 'member'
outputExcelFile = fullfile(outputFolder, strcat(currentFolderName, '_', member, '_Evaluation.xlsx'));

%outputExcelFile = fullfile(outputFolder, strcat(member, '_Evaluation.xlsx'));
xlswrite(outputExcelFile, [headers; excelData], 'Data');

% Write summary statistics to the text file with aligned formatting
summaryFormat = '%-5s %-15s %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f\n';

% Print each summary statistic row to the text file

fprintf(fileID_slice, summaryFormat, 'Max', '', v1_max, y_max, time_max);
fprintf(fileID_slice, summaryFormat, 'Min', '', v1_min, y_min, time_min);
fprintf(fileID_slice, summaryFormat, 'Mean', '', v1_mean, y_mean, time_mean);
fprintf(fileID_slice, summaryFormat, 'Std', '',  v1_std, y_std, time_std);

% Close the text file
fclose(fileID_slice);

disp('Data and summary statistics saved to Excel and text file successfully.');

