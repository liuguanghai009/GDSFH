function net = visiontransformer(Name, image_size, patch_size, in_chans, dim, depth, num_heads, epsilon, nc, ifls, headnum, dlntodan,DROP_PATH_RATE)

if nargin ~= 11

    Name = 'B'; % 规模 T S B L

    image_size = [384,384]; % 图像输入尺寸大小，需要整除 patch_size 以及 window_size

    patch_size = 16; % PatchEmbed 中 二维卷积 stride, kernelsize 的大小，

    in_chans = 3; % 输入通道

    switch Name

        case "T"
            depth = 12; 
            num_heads = 12;
            dim = 192;

        case "S"
            depth = 12;
            num_heads = 12;
            dim = 384;


        case "B"
            depth = 12;
            num_heads = 12;
            dim = 768;

        case "L"
            depth = 24;
            num_heads = 16;
            dim = 1024;

    end

    nc = 2; % 层归一化的维度，这里3 是为了方便得到三维特征图，vit 则是 2

    epsilon = 1e-6; %dinov2 是1e-6,其余是1e-5

    ifls = 1; %是否用layer_scale，一般dinov2用

    headnum = 1000; % in21k : 21841, in1k : 1000, dinov2：1024

    dlntodan = 0; % 是否转为用activations提特征

    DROP_PATH_RATE = 0.1;

end

isreshape = 0;

if ifls == 0

    Mean = [127.50, 127.50, 127.50];
    Std = [127.50, 127.50, 127.50];

else

    Mean = [123.6750, 116.28, 103.53];
    Std = [58.3950, 57.1200, 57.3750];

end

Mean = reshape(Mean,[1,1,3]);

Mean = repmat(Mean,[image_size(1),image_size(2),1]);

Std = reshape(Std,[1,1,3]);

Std = repmat(Std,[image_size(1),image_size(2),1]);

InPut = [imageInputLayer([image_size(1) image_size(2) in_chans],"Name","input","Normalization","zscore",'Mean',Mean,'StandardDeviation',Std);
  ];

lgraph = layerGraph();

lgraph = addLayers(lgraph,InPut);

%%

% PatchEmbed (H,W,3,B) -> (H/4 * W/4,96,B)

layer_pe = [

PatchEmbed('backbone_patch_embed_projection',image_size,patch_size,in_chans,dim,isreshape); % convolution2dLayer(patch_size, embed_dim,"Stride",[patch_size,patch_size],"Name",'patch_embed_proj')

dropoutLayer(0,'Name','pos_drop');

pos_embed_('backbone_pos_embed', dim, image_size, patch_size);

];

lgraph = addLayers(lgraph,layer_pe);

lgraph = connectLayers(lgraph,'input','backbone_patch_embed_projection');

%%

dpr = linspace(0,DROP_PATH_RATE,sum(depth));

for d = 1:depth

    drop_path_ = dpr(d);

    % 设计基本层

    LN = Layernorm_(['backbone_layers_',num2str(d-1),'_ln1'],dim,nc,epsilon); %         LN = layerlnalizationLayer("Name",['backbone_layers_',num2str(j-1),'_ln1'],'Epsilon',1e-5,'OperationDimension','channel-only');

    QKV = qkvCompute_vit(['backbone_layers_',num2str(d-1),'_attn_qkv'], dim, num_heads,nc,drop_path_);

    DP =  DropPath(['backbone_layers_',num2str(d-1),'_attnDropPath'], drop_path_, 1);

    LS = LayerScale(['backbone_layers_',num2str(d-1),'_attn_gamma1'],dim,ifls);

    ATTNadd = additionLayer(2,"Name",['backbone_layers_',num2str(d-1),'_attn_add']);

    ffn = [

    Layernorm_(['backbone_layers_',num2str(d-1),'_ln2'],dim,nc,epsilon); %layerlnalizationLayer("Name",['backbone_layers_',num2str(j-1),'_ln2'],'Epsilon',1e-5,'OperationDimension','channel-only')

    Linear_(['backbone_layers_',num2str(d-1),'_ffn_layers_0_0'],dim,4*dim,nc)

%     geluLayer('Name',['backbone_layers_',num2str(d-1),'_ffn_gelu',num2str(d)],'Approximation','tanh')
    gelu(['backbone_layers_',num2str(d-1),'_ffn_gelu',num2str(d)])

    Linear_(['backbone_layers_',num2str(d-1),'_ffn_layers_1'],4*dim,dim,nc)

    dropoutLayer(0,'Name',['backbone_layers_',num2str(d-1),'_dropout'])

    DropPath(['backbone_layers_',num2str(d-1),'_ffnDropPath'],drop_path_,1);

    LayerScale(['backbone_layers_',num2str(d-1),'_ffn_gamma2'],dim,ifls)

    ];

    FFNadd = additionLayer(2,"Name",['backbone_layers_',num2str(d-1),'_ffn_add']);

    %%

    % 添加层进 lgraph 中

    lgraph = addLayers(lgraph,LN);

    lgraph = addLayers(lgraph,QKV);

    lgraph = addLayers(lgraph, DP);

    lgraph = addLayers(lgraph,LS);

    lgraph = addLayers(lgraph,ATTNadd);

    lgraph = addLayers(lgraph,ffn);

    lgraph = addLayers(lgraph,FFNadd);

    %%

    % 连接 lgraph 中的各个层

    if d == 1

        lgraph = connectLayers(lgraph,'backbone_pos_embed',['backbone_layers_',num2str(d-1),'_ln1']);

        lgraph = connectLayers(lgraph,'backbone_pos_embed',['backbone_layers_',num2str(d-1),'_attn_add/in2']);

    else

        lgraph = connectLayers(lgraph,['backbone_layers_',num2str(d-2),'_ffn_add'],['backbone_layers_',num2str(d-1),'_ln1']);

        lgraph = connectLayers(lgraph,['backbone_layers_',num2str(d-2),'_ffn_add'],['backbone_layers_',num2str(d-1),'_attn_add/in2']);

    end

    lgraph = connectLayers(lgraph,['backbone_layers_',num2str(d-1),'_ln1'],['backbone_layers_',num2str(d-1),'_attn_qkv']);

    lgraph = connectLayers(lgraph,['backbone_layers_',num2str(d-1),'_attn_qkv'],['backbone_layers_',num2str(d-1),'_attn_gamma1']);

    lgraph = connectLayers(lgraph,['backbone_layers_',num2str(d-1),'_attn_gamma1'],['backbone_layers_',num2str(d-1),'_attnDropPath']);

    lgraph = connectLayers(lgraph,['backbone_layers_',num2str(d-1),'_attnDropPath'],['backbone_layers_',num2str(d-1),'_attn_add/in1']);

    % lgraph = connectLayers(lgraph,['backbone_layers_',num2str(d-1),'_attn_gamma1'],['backbone_layers_',num2str(d-1),'_attn_add/in1']);

    lgraph = connectLayers(lgraph,['backbone_layers_',num2str(d-1),'_attn_add'],['backbone_layers_',num2str(d-1),'_ln2']);

    lgraph = connectLayers(lgraph,['backbone_layers_',num2str(d-1),'_attn_add'],['backbone_layers_',num2str(d-1),'_ffn_add/in1']);

    lgraph = connectLayers(lgraph,['backbone_layers_',num2str(d-1),'_ffn_gamma2'],['backbone_layers_',num2str(d-1),'_ffn_add/in2']);

end

%%

%last ln

if ifls == 0
    output = [

    Layernorm_('backbone_ln1',dim,nc,epsilon);

    format_output('out','avgp')

    Linear_('backbone_head',dim,headnum,nc+1)
    ];
    lastname = 'backbone_head';

else
    output = [

    Layernorm_('backbone_ln1',dim,nc,epsilon);

    format_output('out','cls_token')
    ];
    lastname = 'out';
end

lgraph = addLayers(lgraph,output);

lgraph = connectLayers(lgraph,['backbone_layers_',num2str(d-1),'_ffn_add'],'backbone_ln1');

%%

net = dlnetwork(lgraph);

%%

% 导入divnov2  参数
if ifls == 1
    params = importdata(['vit_',Name,'_patch',num2str(patch_size),'_',num2str(image_size(1)),'_dinov2.mat']);
else
    params = importdata(['vit_',Name,'_patch',num2str(patch_size),'_',num2str(image_size(1)),'.mat']);
end
% params = importdata("parameters\dinov3-base.mat");
n = 0;

for i = 1 : height(net.Learnables)

    paramsName = fieldnames(params);

    for j = 1 : numel(paramsName)

        if contains(paramsName{j},net.Learnables.Layer{i,1})  || contains(paramsName{j},"attn_proj") || contains(paramsName{j},"cls_token") || contains(paramsName{j},"pos_embed")

            w = dlarray(params.(paramsName{j}));

            if contains(paramsName{j},'cls_token') &&  contains(net.Learnables.Parameter{i,1},'cls_token')

                w =  reshape(w,size(net.Learnables.Value{i,1}));

            elseif contains(paramsName{j},'pos_embed') &&  contains(net.Learnables.Parameter{i,1},'pos_embed')

                w = permute(w,[2,3,1]);

            elseif contains(paramsName{j},'ln') && contains(paramsName{j},'weight') && contains(net.Learnables.Parameter{i,1},'Scale')

                w =  reshape(w,size(net.Learnables.Value{i,1}));

            elseif contains(paramsName{j},'ln') && contains(paramsName{j},'bias') && contains(net.Learnables.Parameter{i,1},'Offset')

                w =  reshape(w,size(net.Learnables.Value{i,1}));

            elseif contains(paramsName{j},'attn_qkv') && contains(paramsName{j},'weight') && contains(net.Learnables.Parameter{i,1},'qkv/Weights')

                w = permute(w,[2,1]);

            elseif contains(paramsName{j},'attn_proj') && contains(paramsName{j},'weight') && contains(net.Learnables.Parameter{i,1},'proj/Weights')

                w = permute(w,[2,1]);

            elseif contains(paramsName{j},'attn_qkv') && contains(paramsName{j},'bias') && contains(net.Learnables.Parameter{i,1},'qkv/Bias')

                w =  reshape(w,size(net.Learnables.Value{i,1}));

            elseif contains(paramsName{j},'attn_proj') && contains(paramsName{j},'bias') && contains(net.Learnables.Parameter{i,1},'proj/Bias')

                w =  reshape(w,size(net.Learnables.Value{i,1}));

            elseif contains(paramsName{j},'ffn_layers') && contains(paramsName{j},'weight') && contains(net.Learnables.Parameter{i,1},'Weights')

                w = permute(w,[2,1]);

            elseif contains(paramsName{j},'ffn_layers') && contains(paramsName{j},'bias') && contains(net.Learnables.Parameter{i,1},'Bias')

                w =  reshape(w,size(net.Learnables.Value{i,1}));

            elseif contains(paramsName{j},'patch_embed_projection')  && contains(paramsName{j},'weight') && contains(net.Learnables.Parameter{i,1},'Weights')

                w = permute(w,[3,4,2,1]);

            elseif contains(paramsName{j},'patch_embed_projection')  && contains(paramsName{j},'bias') && contains(net.Learnables.Parameter{i,1},'Bias')

                w =  reshape(w,size(net.Learnables.Value{i,1}));

            elseif contains(paramsName{j},'attn_gamma1')  && contains(net.Learnables.Layer{i,1},'attn_gamma1')

                w =  reshape(w,size(net.Learnables.Value{i,1}));

            elseif contains(paramsName{j},'ffn_gamma2')  && contains(net.Learnables.Layer{i,1},'ffn_gamma2')

                w =  reshape(w,size(net.Learnables.Value{i,1}));

            elseif contains(paramsName{j},'backbone_ln1_weight')  && contains(net.Learnables.Parameter{i,1},'Scale')

                w =  reshape(w,size(net.Learnables.Value{i,1}));

            elseif contains(paramsName{j},'backbone_ln1_bias')  && contains(net.Learnables.Parameter{i,1},'Offset')

                w =  reshape(w,size(net.Learnables.Value{i,1}));

            elseif contains(paramsName{j},'backbone_head_weight')  && contains(net.Learnables.Parameter{i,1},'Weights') && contains(net.Learnables.Layer{i,1},'backbone_head')

                w =  reshape(w,size(net.Learnables.Value{i,1}));

            elseif contains(paramsName{j},'backbone_head_bias')  && contains(net.Learnables.Parameter{i,1},'Bias') && contains(net.Learnables.Layer{i,1},'backbone_head')

                w =  reshape(w,size(net.Learnables.Value{i,1}));

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
if ifls == 1
    fprintf('导入参数情况 %d / %d\n',n,height(net.Learnables))
else
    fprintf('导入参数情况 %d / %d\n',n,height(net.Learnables)-2*depth)
end

if dlntodan == 1
    lgraph = layerGraph(net);
    
    sc = [
    softmaxLayer("Name",'sfm')
    
    classificationLayer('Classes',categorical(1:headnum))];
    
    lgraph = addLayers(lgraph,sc);
    
    lgraph = connectLayers(lgraph,lastname,'sfm');
    
    net = assembleNetwork(lgraph);
end

end