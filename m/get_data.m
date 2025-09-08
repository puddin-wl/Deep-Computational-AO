mainFolder1='E:\DIP_shuju\noise\dip';%gt
mainFolder2='E:\DIP_shuju\noise\dip1';%wf
folder1 = fullfile(mainFolder1);
folder2 = fullfile(mainFolder2);

% 获取两个文件夹中的图像文件
images1 = dir(fullfile(folder1, '*.tif')); % 假设是tiff格式
images2 = dir(fullfile(folder2, '*.tif'));

% 检查图像数量是否相同
if length(images1) ~= length(images2)
    error('两个子文件夹中的图像数量不匹配');
end

% 创建新的主文件夹
newMainFolder = fullfile('E:\DL_SR\data\noise');
if ~exist(newMainFolder, 'dir')
    mkdir(newMainFolder);
end

% 处理每对图像
for i = 1:length(images1)
    % 创建编号子文件夹 (0001, 0002, ...)
    subFolderName = sprintf('%04d', i);
    newSubFolder = fullfile(newMainFolder, subFolderName);
    if ~exist(newSubFolder, 'dir')
        mkdir(newSubFolder);
    end
    
    % 复制第一个图像并重命名为00.tiff
    img1Path = fullfile(folder1, images1(i).name);
    copyfile(img1Path, fullfile(newSubFolder, '00.tiff'));
    
    % 复制第二个图像并重命名为01.tiff
    img2Path = fullfile(folder2, images2(i).name);
    copyfile(img2Path, fullfile(newSubFolder, '01.tiff'));
end