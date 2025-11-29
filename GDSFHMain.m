function GDSFHMain

datasets = ["Oxford_5K库","Paris_6K库","Oxford_105K库","Paris_106K库","Holidays_upright库","UKB库","ROxford5k库","RParis6k库","GHIM_10K库","Corel_10K库"];

net_name = ["vgg16","resnet101","repvgg","retraining_vgg","retraining_ResNet101","retraining_repvgg","ViT"];

layerList = ["relu5_3","res5c_relu","relu4_24","relu5_3","res5c_relu","relu4_24","backbone_ln1"];

numList = [512,2048,640,512,2048,640,1024];

%选择数据库
index = 1;

dataname = datasets(index);

%选择网络
number = 5;

netname = net_name(number);

% 自动确定某层的特征图数量

layer = layerList(number);

num = numList(number);
%
if strcmp(netname,'repvgg')
    %
    net = importdata('G:/GDSFH/package/imagenet_repvggplus_L2pse_deploy.mat');
    %
elseif strcmp(netname,'retraining_vgg')
    %
    net =  importdata('G:/GDSFH/package/VGG16-epoch50-size448-dim512.mat');
    %
elseif strcmp(netname,'retraining_ResNet101')
    %
    net = importdata('G:/GDSFH/package/ResNet101-epoch50-size448-dim2048.mat');
    %
elseif strcmp(netname,'retraining_repvgg')
    %
    net = importdata('G:/GDSFH/package/RepVGG-epoch50-size448-dim640.mat');
    %
elseif strcmp(netname,'ViT')
    %
    net = importdata('G:/GDSFH/package/VIT_Large_dim1024.mat');
    %
else
    net = eval(netname);
end
%
filepatch = "F:/标准图像库/" + dataname + "/图库/";

filename = dir(filepatch + "*.jpg");

[file_num, ~] = size(filename);

%

positiveCrow = zeros(file_num,num);
negativeCrow = zeros(file_num,num);

% rawData不能再改动,否则超像素分割得到的是白色结果

for i = 1:file_num

    im = imread("F:/标准图像库/" + dataname + "/图库/" + filename(i).name);

    if size(im,3)==1
        rgb = cat(3,im,im,im);
        im = mat2gray(rgb);
    end
    %
    rawData = im;

    im = single(im);
    %
    [h, w, ~] = size(im);

    %
    if strcmp(netname,'ViT')
        img_resize = imresize(im, [512 512]);
        img_auge = imresize(im, [224 224]);

    elseif contains(netname,"dinov3")
        img_resize = imresize(im, [512 512],'bilinear');
        img_auge = imresize(im, [224 224],'bilinear');
    else
        if dataname == "UKB库"
            img_resize = im;
        else
            if(h < w)
                img_resize = imresize(im, [768 1024]);
            else
                img_resize = imresize(im, [1024 768]);
            end
        end
        %
        img_auge = imresize(im, [224 224]);
    end

    if contains(netname, ["dinov3", "ViT"])

        img_auge = dlarray(gpuArray(img_auge), "SSCB");
        img_resize = dlarray(gpuArray(img_resize), "SSCB");

        negativepool5 = predict(net, img_auge, "Outputs", layer);
        pool5 = predict(net, img_resize, "Outputs", layer);

        negativepool5 = gather(extractdata(relu(negativepool5)));
        pool5 = gather(extractdata(relu(pool5)));

        if contains(netname, "dinov3")
            negativepool5(2:5, :) = [];
            pool5(2:5, :) = [];
        end
        %
        [negativepool5, clsTokenNegativepool5] = RerangeToFeaturemaps(negativepool5);

        [pool5, clsTokenPool5] = RerangeToFeaturemaps(pool5);

    else
        negativepool5 = activations(net, img_auge, layer, 'OutputAs', 'channels');
        pool5 = activations(net, img_resize, layer, 'OutputAs', 'channels');

    end
    %
    [positiveCrow(i,:),negativeCrow(i,:)] = GDSFH_Representation(pool5,negativepool5,rawData);

    %
    if (mod(i,5)==0)
        fprintf('i=%d\n',i);
    end
end


positive_deepCrow = normalize(positiveCrow,2,'norm');

negative_deepCrow = normalize(negativeCrow,2,'norm');

for i=1:file_num

    split = strsplit(filename(i).name, {'.'});
    name = split(1);

    %%
    positive_feature_save = positive_deepCrow(i,:)';
    save(['F:/标准图像库/',dataname{1},'/Matlab提取的特征/DOFH/正例/',name{1},'.txt'],'positive_feature_save','-ASCII');
    %%
    negative_feature_save = negative_deepCrow(i,:)';
    save(['F:/标准图像库/',dataname{1},'/Matlab提取的特征/DOFH/负例/',name{1},'.txt'],'negative_feature_save','-ASCII');

    %
    if (mod(i,5)==0)
        fprintf('i=%d\n',i);
    end
end
%
end
