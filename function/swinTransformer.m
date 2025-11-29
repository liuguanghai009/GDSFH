function net = swinTransformer(Name,image_size, patch_size, in_chans, window_size,nc,epsilon, depths, num_heads, embed_dim, headnum, dlntodan,DROP_PATH_RATE)

if nargin ~= 12

    Name = 'B';

    image_size = [384,384];

    patch_size = 4;

    in_chans = 3;

    window_size = 12;

    nc = 2;

    epsilon = 1e-5;

    switch Name

        case "T"
            depths = [2,2,6,2];
            num_heads = [3,6,12,24];
            embed_dim = 96;

        case "S"
            depths = [2,2,18,2];
            num_heads = [3,6,12,24];
            embed_dim = 96;

        case "B"
            depths = [2,2,18,2];
            num_heads = [4,8,16,32];
            embed_dim = 128;

        case "L"
            depths = [2,2,18,2];
            num_heads = [6,12,24,48];
            embed_dim = 192;

    end

    headnum = 21841; % in21k : 21841, in1k : 1000

    dlntodan = 0;

    DROP_PATH_RATE = 0.1;

end

%% 导入参数路径

params = importdata(['vision_Transformer/parameters/swin_',Name,'_patch',num2str(patch_size),'_window',num2str(window_size),'_',num2str(image_size(1)),'_im22k.mat']);

%% 输入层设置
Mean = [123.6750, 116.28, 103.53];

Mean = reshape(Mean,[1,1,3]);

Mean = repmat(Mean,[image_size(1),image_size(2),1]);

Std = [58.3950, 57.1200, 57.3750];

Std = reshape(Std,[1,1,3]);

Std = repmat(Std,[image_size(1),image_size(2),1]);

InPut = imageInputLayer([image_size(1) image_size(2) in_chans],"Name","input","Normalization","zscore",'Mean',Mean,'StandardDeviation',Std);

lgraph = layerGraph();

lgraph = addLayers(lgraph,InPut);

%%

% PatchEmbed (H,W,3,B) -> (H/4 * W/4,96,B)

isreshape = 1;

layer_pe = [

    PatchEmbed('patch_embed_proj',image_size,patch_size,in_chans,embed_dim,isreshape); % convolution2dLayer(patch_size, embed_dim,"Stride",[patch_size,patch_size],"Name",'patch_embed_proj')
    
    Layernorm_('patch_embed_norm',embed_dim,nc,epsilon);% layerNormalizationLayer("Name",'patch_embed_norm','Epsilon',1e-5,'OperationDimension','channel-only');
    
    dropoutLayer(0,'Name','pos_drop');

];

lgraph = addLayers(lgraph,layer_pe);

lgraph = connectLayers(lgraph,'input','patch_embed_proj');

%%

% BasicLayer

% dim = embed_dim*(2^i_layer)

% 每个patch的resolution

patch_resolution(1) = image_size(1) / patch_size;

patch_resolution(2) = image_size(2) / patch_size;

dpr = linspace(0,DROP_PATH_RATE,sum(depths));

for i = 1:numel(depths)

    dim = (embed_dim*(2^(i-1)));

    input_resolution = [fix( patch_resolution(1) / 2^(i-1)),fix( patch_resolution(2) /2^(i-1))];

    depth = depths(i);

    num_head = num_heads(i);

    if i == 1

        drop_path = dpr(1:sum(depths(1:i)));

    else

        drop_path = dpr(sum(depths(1:i-1))+1:sum(depths(1:i)));

    end

    for j = 1:depth

        shift_size = (mod(j, 2) ~= 0) * 0 + (mod(j, 2) == 0) * fix(window_size / 2);

        drop_path_ = drop_path(j);

        if min(input_resolution) <= window_size

            shift_size = 0;

            window_size = min(input_resolution);

        end

%%        

% 设计基本层

        LN = Layernorm_(['layers_',num2str(i-1),'_blocks_',num2str(j-1),'_norm1'],dim,nc,epsilon); %         LN = layerNormalizationLayer("Name",['layers_',num2str(i-1),'_blocks_',num2str(j-1),'_norm1'],'Epsilon',1e-5,'OperationDimension','channel-only');

        QKV = qkvCompute_swin(['layers_',num2str(i-1),'_blocks_',num2str(j-1),'_attn_qkv'], dim, window_size, num_head, shift_size, input_resolution,nc,drop_path_);

        DP =  DropPath(['layers_',num2str(i-1),'_blocks_',num2str(j-1),'_attnDropPath'], drop_path_, 1);

        ATTNadd = additionLayer(2,"Name",['layers_',num2str(i-1),'_blocks_',num2str(j-1),'_attn_add']);

        MLP = [

            Layernorm_(['layers_',num2str(i-1),'_blocks_',num2str(j-1),'_norm2'],dim,nc,epsilon); %layerNormalizationLayer("Name",['layers_',num2str(i-1),'_blocks_',num2str(j-1),'_norm2'],'Epsilon',1e-5,'OperationDimension','channel-only')
    
            Linear_(['layers_',num2str(i-1),'_blocks_',num2str(j-1),'_mlp_fc1'],dim,4*dim,nc)
    
%             geluLayer('Name',['layers_',num2str(i-1),'_blocks_',num2str(j-1),'_mlp_gelu',num2str(j)],'Approximation','tanh')

            gelu(['layers_',num2str(i-1),'_blocks_',num2str(j-1),'_mlp_gelu',num2str(j)])
    
            Linear_(['layers_',num2str(i-1),'_blocks_',num2str(j-1),'_mlp_fc2'],4*dim,dim,nc)
    
            dropoutLayer(0,'Name',['layers_',num2str(i-1),'_blocks_',num2str(j-1),'_dropout'])
    
            DropPath(['layers_',num2str(i-1),'_blocks_',num2str(j-1),'_mlpDropPath'],drop_path_,1);

        ];

        FFNadd = additionLayer(2,"Name",['layers_',num2str(i-1),'_blocks_',num2str(j-1),'_ffn_add']);

%%

% 添加层进 lgraph 中

        lgraph = addLayers(lgraph,LN);

        lgraph = addLayers(lgraph,QKV);

        lgraph = addLayers(lgraph, DP);

        lgraph = addLayers(lgraph,ATTNadd);

        lgraph = addLayers(lgraph,MLP);

        lgraph = addLayers(lgraph,FFNadd);

%%

% 连接 lgraph 中的各个层

        if i == 1 && j == 1

            lgraph = connectLayers(lgraph,'pos_drop',['layers_',num2str(i-1),'_blocks_',num2str(j-1),'_norm1']);

            lgraph = connectLayers(lgraph,'pos_drop',['layers_',num2str(i-1),'_blocks_',num2str(j-1),'_attn_add/in2']);

        elseif(j == 1)

            lgraph = connectLayers(lgraph,['layers_',num2str(i-2),'_downsample_reduction'],['layers_',num2str(i-1),'_blocks_',num2str(j-1),'_norm1']);

            lgraph = connectLayers(lgraph,['layers_',num2str(i-2),'_downsample_reduction'],['layers_',num2str(i-1),'_blocks_',num2str(j-1),'_attn_add/in2']);

        else

            lgraph = connectLayers(lgraph,['layers_',num2str(i-1),'_blocks_',num2str(j-2),'_ffn_add'],['layers_',num2str(i-1),'_blocks_',num2str(j-1),'_norm1']);            

            lgraph = connectLayers(lgraph,['layers_',num2str(i-1),'_blocks_',num2str(j-2),'_ffn_add'],['layers_',num2str(i-1),'_blocks_',num2str(j-1),'_attn_add/in2']);

        end

        lgraph = connectLayers(lgraph,['layers_',num2str(i-1),'_blocks_',num2str(j-1),'_norm1'],['layers_',num2str(i-1),'_blocks_',num2str(j-1),'_attn_qkv']);

        lgraph = connectLayers(lgraph,['layers_',num2str(i-1),'_blocks_',num2str(j-1),'_attn_qkv'],['layers_',num2str(i-1),'_blocks_',num2str(j-1),'_attnDropPath']);
        
        lgraph = connectLayers(lgraph,['layers_',num2str(i-1),'_blocks_',num2str(j-1),'_attnDropPath'],['layers_',num2str(i-1),'_blocks_',num2str(j-1),'_attn_add/in1']);

%         lgraph = connectLayers(lgraph,['layers_',num2str(i-1),'_blocks_',num2str(j-1),'_attn_qkv'],['layers_',num2str(i-1),'_blocks_',num2str(j-1),'_attn_add/in1']);

        lgraph = connectLayers(lgraph,['layers_',num2str(i-1),'_blocks_',num2str(j-1),'_attn_add'],['layers_',num2str(i-1),'_blocks_',num2str(j-1),'_norm2']);

        lgraph = connectLayers(lgraph,['layers_',num2str(i-1),'_blocks_',num2str(j-1),'_attn_add'],['layers_',num2str(i-1),'_blocks_',num2str(j-1),'_ffn_add/in1']);

%         lgraph = connectLayers(lgraph,['layers_',num2str(i-1),'_blocks_',num2str(j-1),'_dropout'],['layers_',num2str(i-1),'_blocks_',num2str(j-1),'_ffn_add/in2']);
        lgraph = connectLayers(lgraph,['layers_',num2str(i-1),'_blocks_',num2str(j-1),'_mlpDropPath'],['layers_',num2str(i-1),'_blocks_',num2str(j-1),'_ffn_add/in2']);


    end

%% 

% 若不是最后一个 BasicBlock，则需要 patchMerging 

    if i < 4

        PM = [

        PatchMerging(['layers_',num2str(i-1),'_downsample'],input_resolution,dim)

        Layernorm_(['layers_',num2str(i-1),'_downsample_norm'],4*dim,nc,epsilon); %         layerNormalizationLayer("Name",['layers_',num2str(i-1),'_downsample_norm'],'Epsilon',1e-5,'OperationDimension','channel-only');

        Linear_(['layers_',num2str(i-1),'_downsample_reduction'],4*dim,2*dim,nc)

        ];

        lgraph = addLayers(lgraph,PM);

        lgraph = connectLayers(lgraph,['layers_',num2str(i-1),'_blocks_',num2str(j-1),'_ffn_add'],['layers_',num2str(i-1),'_downsample']);

    end

end

%%

%last norm & fc

final = [

Layernorm_('norm',dim,nc,epsilon);

format_output('out','avgp')

Linear_('head',dim,headnum,nc+1);

];

lgraph = addLayers(lgraph,final);

lgraph = connectLayers(lgraph,['layers_',num2str(i-1),'_blocks_',num2str(j-1),'_ffn_add'],'norm');

net = dlnetwork(lgraph);

%%

% 导入预训练参数

n = 0;

for i = 1 : height(net.Learnables)

    paramsName = fieldnames(params);

    for j = 1 : numel(paramsName)

        if contains(paramsName{j},net.Learnables.Layer{i,1}) || contains(net.Learnables.Parameter{i,1},"relative_position_bias_table") || contains(paramsName{j},"attn_proj")

            w = dlarray(params.(paramsName{j}));

            if contains(paramsName{j},'relative_position_bias_table')

                w =  reshape(w,size(net.Learnables.Value{i,1}));

            elseif (contains(paramsName{j},'norm1_weight') || contains(paramsName{j},'norm2_weight')) && contains(net.Learnables.Parameter{i,1},'Scale')

                w =  reshape(w,size(net.Learnables.Value{i,1}));

            elseif (contains(paramsName{j},'norm1_bias') || contains(paramsName{j},'norm2_bias')) && contains(net.Learnables.Parameter{i,1},'Offset')

                w =  reshape(w,size(net.Learnables.Value{i,1}));

            elseif contains(paramsName{j},'attn_qkv_weight') && contains(net.Learnables.Parameter{i,1},'qkv/Weights')

                w = permute(w,[2,1]);

            elseif contains(paramsName{j},'attn_proj_weight') && contains(net.Learnables.Parameter{i,1},'proj/Weights')

                w = permute(w,[2,1]);

            elseif contains(paramsName{j},'attn_qkv_bias') && contains(net.Learnables.Parameter{i,1},'qkv/Bias')

                w =  reshape(w,size(net.Learnables.Value{i,1}));

            elseif contains(paramsName{j},'attn_proj_bias') && contains(net.Learnables.Parameter{i,1},'proj/Bias')

                w =  reshape(w,size(net.Learnables.Value{i,1}));

            elseif (contains(paramsName{j},'mlp_fc1_weight') || contains(paramsName{j},'mlp_fc2_weight')) && contains(net.Learnables.Parameter{i,1},'Weights')

                w = permute(w,[2,1]);

            elseif (contains(paramsName{j},'mlp_fc1_bias') || contains(paramsName{j},'mlp_fc2_bias')) && contains(net.Learnables.Parameter{i,1},'Bias')

                w =  reshape(w,size(net.Learnables.Value{i,1}));

            elseif contains(paramsName{j},'downsample_norm_weight')  && contains(net.Learnables.Parameter{i,1},'Scale')

                w =  reshape(w,size(net.Learnables.Value{i,1}));

            elseif contains(paramsName{j},'downsample_norm_bias')  && contains(net.Learnables.Parameter{i,1},'Offset')

                w =  reshape(w,size(net.Learnables.Value{i,1}));

            elseif contains(paramsName{j},'downsample_reduction')  && contains(net.Learnables.Parameter{i,1},'Weights')

                w = permute(w,[2,1]);

            elseif contains(paramsName{j},'patch_embed_proj_weight') && contains(net.Learnables.Parameter{i,1},'Weights')

                w = permute(w,[3,4,2,1]);

            elseif contains(paramsName{j},'patch_embed_proj_bias') && contains(net.Learnables.Parameter{i,1},'Bias')

                w =  reshape(w,size(net.Learnables.Value{i,1}));

            elseif contains(paramsName{j},'patch_embed_norm_weight')  && contains(net.Learnables.Parameter{i,1},'Scale')

                w =  reshape(w,size(net.Learnables.Value{i,1}));

            elseif contains(paramsName{j},'patch_embed_norm_bias')  && contains(net.Learnables.Parameter{i,1},'Offset')

                w =  reshape(w,size(net.Learnables.Value{i,1}));

            elseif contains(paramsName{j},'norm_weight')  && contains(net.Learnables.Parameter{i,1},'Scale') && contains(net.Learnables.Layer{i,1},'norm')

                w =  reshape(w,size(net.Learnables.Value{i,1}));

            elseif contains(paramsName{j},'norm_bias')  && contains(net.Learnables.Parameter{i,1},'Offset') && contains(net.Learnables.Layer{i,1},'norm')

                w =  reshape(w,size(net.Learnables.Value{i,1}));

            elseif contains(paramsName{j},'head_weight')  && contains(net.Learnables.Layer{i,1},'head') && contains(net.Learnables.Parameter{i,1},'Weights')

                w =  reshape(w,size(net.Learnables.Value{i,1}));

            elseif contains(paramsName{j},'head_bias')  && contains(net.Learnables.Layer{i,1},'head') && contains(net.Learnables.Parameter{i,1},'Bias')

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

fprintf('导入参数情况 %d / %d\n',n,height(net.Learnables)-3) % layers_?_downsample_reduction bias 为 0 不需要导入

if dlntodan == 1
    lgraph = layerGraph(net);
    
    sc = [
    softmaxLayer("Name",'sfm')
    
    classificationLayer('Classes',categorical(1:headnum))];
    
    lgraph = addLayers(lgraph,sc);
    
    lgraph = connectLayers(lgraph,'head','sfm');
    
    net = assembleNetwork(lgraph);
end

end