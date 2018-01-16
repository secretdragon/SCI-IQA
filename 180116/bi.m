% weight=zeros(295764,1);
for i = 1:295764
    if i<10
        name=strcat('E:/iqa/cnnpy/DTA/000000',num2str(i),'.jpg');
    elseif i <100
        name=strcat('E:/iqa/cnnpy/DTA/00000',num2str(i),'.jpg');
    elseif i <1000
        name=strcat('E:/iqa/cnnpy/DTA/0000',num2str(i),'.jpg');
    elseif i <10000
        name=strcat('E:/iqa/cnnpy/DTA/000',num2str(i),'.jpg');  
    elseif i <100000
        name=strcat('E:/iqa/cnnpy/DTA/00',num2str(i),'.jpg');  
    elseif i <231085
        name=strcat('E:/iqa/cnnpy/DTA/0',num2str(i),'.jpg');  
    else 
        name=strcat('E:/iqa/cnnpy/DTA/0',num2str(i),'.jpg');
    end
    img=imread(name,'jpg');
    img=imresize(img,2.4);
    name2=strrep(name,'DTA','DTN');
    imwrite(img,name2,'jpg');
% img=img.*255;
%     img=rgb2gray(img);
%     img=double(img);
%     a=calcSVD7(img);
%     weight(i)=a;
end
