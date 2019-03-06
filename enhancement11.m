clear all
close all
path='K:\Matlab_yyz\Warehouse\MATLAB\ZhouKang\Cheng\';
filters=[path,'*.png'];
filelist=dir(filters);

for f=1:length(filelist)
    filestr=filelist(f).name;
    I=imread(filestr);
%     im=double(rgb2gray(mat2gray(I)));
    im=double((mat2gray(I)));
    X=imadjust(im,[0.3, 1 ]); %% BOE: 0.1;   Challenge: 0.15; Cheng:0.3
    [pathstr,name,ext] = fileparts(filelist(f).name);
    I82=strcat(name,'_yyz.tif');
    imwrite(X,strcat('K:\Matlab_yyz\Warehouse\MATLAB\ZhouKang\BOE\',I82))

end


