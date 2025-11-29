function Cross_whitening

Tests = ["Oxford_5K库","Paris_6K库","Oxford_105K库","Paris_106K库"];
Trains = ["Paris_6K库","Oxford_5K库","Paris_6K库","Oxford_5K库"];

%% 选择数据库

Index = 1;

dataXTest = Tests(Index);

dataXTrain = Trains(Index);

%% 网络模型+自动确定某层的特征图数量

number = 5;

net_name = ["vgg16","resnet101","repvgg","retraining_vgg","retraining_ResNet101","retraining_repvgg","Transformer"];

numList = [512,2048,640,512,2048,640,1024];

num = numList(number);

%

fileXTest = dir("F:/标准图像库/" + dataXTest + "/Matlab提取的特征/DOFH/正例/" + "*.txt");

filesXTrain = dir("F:/标准图像库/" + dataXTrain + "/Matlab提取的特征/DOFH/负例/" + "*.txt");
%
[file_num, ~] = size(fileXTest);

[file_count, ~] = size(filesXTrain);

%
positive_deepCrow = zeros(file_num,num);

negative_deepCrow  = zeros(file_count,num);

parfor i=1:file_num

    positive_deepCrow(i,:) = importdata("F:/标准图像库/" + dataXTest{1} + "/Matlab提取的特征/DOFH/正例/" + fileXTest(i).name);

    if (mod(i,5)==0)
        fprintf('i=%d\n',i);
    end

end

parfor i=1:file_count

    negative_deepCrow(i,:) = importdata("F:/标准图像库/" + dataXTrain{1} + "/Matlab提取的特征/DOFH/负例/" + filesXTrain(i).name);

    if (mod(i,5)==0)
        fprintf('i=%d\n',i);
    end

end

dim = 8;

X = transfer_learning_whitening(negative_deepCrow,positive_deepCrow,dim);

for i=1:file_num

    % 获取图片序号d

    split = strsplit(fileXTest(i).name, {'.'});

    name = split(1);

    %%

    feature_save = X(i,:)';

    save(['F:/标准图像库/',dataXTest{1},'/Matlab提取的特征/DOFH/白化特征/',name{1},'.txt'],'feature_save','-ASCII');

    %%
end
