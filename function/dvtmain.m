image_size = [384,384];

Name  = 'dvt';

mName = 'l';

in_chans  = 3;

num_classes  = 1000;

switch mName

    case 's'

        stem_hidden_dim  = 32;

        embed_dims  = [64, 128, 320, 448];

        num_heads  = [2, 4, 10, 14];

        mlp_ratios  = [8, 8, 4, 3, 2];

        depths  = [3, 4, 6, 3];

    case 'b'

        stem_hidden_dim = 64;

        embed_dims = [64, 128, 320, 512];

        num_heads = [2, 4, 10, 16];

        mlp_ratios = [8, 8, 4, 3, 2];
  
        depths = [3, 4, 15, 3];

    case 'l'

        stem_hidden_dim = 64;

        embed_dims = [96, 192, 384, 512];

        num_heads = [3, 6, 12, 16];

        mlp_ratios = [8, 8, 4, 3, 2];

        depths = [3, 6, 21, 3];

end

drop_path_rate  = 0;

num_stages  = 4;

sep_stages  = 2;

token_label  = 1;

image_Size  = 384;

Mean = [123.6750, 116.28, 103.53];

Mean = reshape(Mean,[1,1,3]);

Mean = repmat(Mean,[image_size(1),image_size(2),1]);

Std = [58.3950, 57.1200, 57.3750];

Std = reshape(Std,[1,1,3]);

Std = repmat(Std,[image_size(1),image_size(2),1]);

InPut = imageInputLayer([image_size(1) image_size(2) in_chans],"Name","input","Normalization","zscore",'Mean',Mean,'StandardDeviation',Std);

% InPut = imageInputLayer([image_size(1) image_size(2) in_chans],"Name","input","Normalization","none");

lgraph = layerGraph();

lgraph = addLayers(lgraph,InPut);

Name1 = 'patch_embed1';

out_channels = embed_dims(1);

conv = [

    convolution2dLayer(7,stem_hidden_dim,'Stride',2,'Padding',3,'BiasInitializer','zeros','BiasLearnRateFactor',0,'Name',[Name1,'_conv_0'],'WeightsInitializer','he');
    
    batchNormalizationLayer('Name',[Name1,'_conv_1'],'Epsilon',1e-5);
    
    reluLayer('Name',[Name1,'_relu_1']);
    
    convolution2dLayer(3,stem_hidden_dim,'Stride',1,'Padding',1,'BiasInitializer','zeros','BiasLearnRateFactor',0,'Name',[Name1,'_conv_3'],'WeightsInitializer','he');
    
    batchNormalizationLayer('Name',[Name1,'_conv_4'],'Epsilon',1e-5);
    
    reluLayer('Name',[Name1,'_relu_2']);
    
    convolution2dLayer(3,stem_hidden_dim,'Stride',1,'Padding',1,'BiasInitializer','zeros','BiasLearnRateFactor',0,'Name',[Name1,'_conv_6'],'WeightsInitializer','he');
    
    batchNormalizationLayer('Name',[Name1,'_conv_7'],'Epsilon',1e-5);
    
    reluLayer('Name',[Name1,'_relu_3']);
    
    convolution2dLayer(3,out_channels,'Stride',2,'Padding',1,'Name',[Name1,'_proj'],'WeightsInitializer','he')

];


dvt = DVT(Name,in_chans,num_classes,stem_hidden_dim,embed_dims,num_heads,mlp_ratios,drop_path_rate,depths,num_stages,sep_stages,token_label,image_Size);

lgraph  = addLayers(lgraph,conv);

lgraph  = addLayers(lgraph,dvt);

lgraph = connectLayers(lgraph,'input',[Name1,'_conv_0']);

lgraph = connectLayers(lgraph,[Name1,'_proj'],'dvt');

net = dlnetwork(lgraph);

%%
% 导入预训练参数

n = 0;

params = importdata(['vision_Transformer\parameters\dualvit_',mName,'_',num2str(image_size(1)),'.mat']);

net_param = {};
for i = 1 : height(net.Learnables)
    parts = split(net.Learnables.Parameter{i,1},'/');
    if numel(parts) ~= 1

        lp = parts{end};
        if numel(parts) ~= 1

            sp = parts{end-1};

        end

        if strcmp(lp,"Weights") || strcmp(lp,"Scale")

            net_param{i} = [sp,'_weight'];

        elseif strcmp(lp,"Bias") || strcmp(lp,"Offset")

            net_param{i} = [sp,'_bias'];

        elseif contains(lp,"gamma")

            net_param{i} = [sp,'_',lp];

        else

            net_param{i} = lp;

        end


    elseif strcmp(net.Learnables.Parameter{i,1},"q")

        net_param{i} = "q";


    else
        if  strcmp(net.Learnables.Parameter{i,1},"Weights") || strcmp(net.Learnables.Parameter{i,1},"Scale")
            net_param{i} = [net.Learnables.Layer{i},'_weight'];
        elseif strcmp(net.Learnables.Parameter{i,1},"Bias") || strcmp(net.Learnables.Parameter{i,1},"Offset")
            net_param{i} = [net.Learnables.Layer{i},'_bias'];
        end

    end

end

net_param = net_param';

%%
for i = 1 : height(net.Learnables)

    paramsName = fieldnames(params);

    for j = 1 : numel(paramsName)

        if strcmp(paramsName{j},net_param{i})

            w = dlarray(params.(paramsName{j}));

            if size(w,1) == 1

                w =  reshape(w,size(net.Learnables.Value{i,1}));

            elseif contains(paramsName{j},"dwconv_dwconv_weight")

                w = permute(w,[3,4,2,5,1]);

            elseif contains(paramsName{j},"dwconv_dwconv_bias")

                w = permute(w,[3,4,1,2]);

            elseif numel(size(w)) == 4

                w =  permute(w,[3,4,2,1]);

            elseif size(w,1) ~= 1

                w = permute(w,[2,1]);

            else

                continue;

            end

        else

            continue;
        end

        params = rmfield(params, paramsName{j});

        net.Learnables.Value{i,1} = w;

        n = n + 1;

        break;

    end

end

for i = 1 : height(net.State)

    paramsName = fieldnames(params);

    for j = 1 : numel(paramsName)

        if contains(paramsName{j},net.State.Layer{i,1})

            w = dlarray(params.(paramsName{j}));

            if contains(paramsName{j},'patch_embed1_conv_1_running_mean')  && contains(net.State.Parameter{i,1},'TrainedMean')

                w =  reshape(w,size(net.State.Value{i,1}));

            elseif contains(paramsName{j},'patch_embed1_conv_1_running_var')  && contains(net.State.Parameter{i,1},'TrainedVariance')

                w =  reshape(w,size(net.State.Value{i,1}));

            elseif contains(paramsName{j},'patch_embed1_conv_4_running_mean')  && contains(net.State.Parameter{i,1},'TrainedMean')

                w =  reshape(w,size(net.State.Value{i,1}));

            elseif contains(paramsName{j},'patch_embed1_conv_4_running_var')  && contains(net.State.Parameter{i,1},'TrainedVariance')

                w =  reshape(w,size(net.State.Value{i,1}));

            elseif contains(paramsName{j},'patch_embed1_conv_7_running_mean')  && contains(net.State.Parameter{i,1},'TrainedMean')

                w =  reshape(w,size(net.State.Value{i,1}));

            elseif contains(paramsName{j},'patch_embed1_conv_7_running_var')  && contains(net.State.Parameter{i,1},'TrainedVariance')

                w =  reshape(w,size(net.State.Value{i,1}));


            else

                continue;

            end

        else

            continue;
        end

        params = rmfield(params, paramsName{j});

        net.State.Value{i,1} = w;

        n = n + 1;

        break;

    end

end
