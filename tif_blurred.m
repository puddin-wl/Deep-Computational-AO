clear; clc;
addpath(genpath('.\mFunc\'));


inputFolder = '';       
outputFolder = '';      
coefficientsFile = 'coeffs.csv'; 

if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end


lambda = 0.532;           
pixelSize = 3;            
pIn = 4:10;               
aValue = 0.45;              
zernType = 'random1';     
NA = 0.1;                 


fileList = dir(fullfile(inputFolder, '*.tif'));
numFiles = length(fileList);


header = ['文件名,' sprintf('coeff%d,', pIn)];
header(end) = []; 
fid = fopen(fullfile(outputFolder, coefficientsFile), 'w');
fprintf(fid, '%s\n', header);


for i = 1:numFiles
   
    fileName = fileList(i).name;
    filePath = fullfile(inputFolder, fileName);
    
    
    img = imread(filePath);
    img = double(img);
    if size(img,3) == 3
        img = rgb2gray(img);  
    end
    [Sx, Sy] = size(img);
    
    
    coeffs = gen_zern_coeffs(pIn, aValue, zernType);

    
    
    
    psf = coeffs2PSF(pIn, coeffs, Sx, pixelSize, lambda, NA);
    

    conv_result = conv2(img,psf,'same');
    

    outputPath = fullfile(outputFolder, fileName);
    imwrite(uint16(65535 * mat2gray(abs(conv_result))), outputPath);
    

    coeffStr = sprintf('%.6f,', coeffs);
    coeffStr(end) = []; %
    

    fprintf(fid, '%s,%s\n', fileName, coeffStr);
    

    fprintf('done %d/%d : %s\n', i, numFiles, fileName);
end


fclose(fid);
disp('== Done ==');
