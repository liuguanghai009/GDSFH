classdef qkvCompute_swinv2 < nnet.layer.Layer %Window based multi-head self attention (W-MSA) module with relative position bias

    properties(Learnable)

        qkvw

        q_bias

        v_bias

        proj

        cpb_mlp

        logit_scale

        attn_drop

        proj_drop

    end

    properties

        dim

        num_head

        relative_position_index

        shift_size

        window_size

        input_resolution

        attn_mask

        k_bias

        relative_coords_table

    end

    methods

        function layer = qkvCompute_swinv2(Name,dim,window_size,num_head,shift_size,input_resolution,nc,pretrained_window_size,drop_path_)

            arguments

                Name

                dim

                window_size

                num_head

                shift_size

                input_resolution

                nc

                pretrained_window_size

                drop_path_

            end

            layer.input_resolution = input_resolution;

            layer.dim = dim;

            layer.Name = Name;

            layer.Description = "qkvCompute_swinv2";

            layer.num_head = num_head;

            layer.window_size = window_size;

            layer.shift_size = shift_size;

            %%
            layer.qkvw  = dlarray(initializeWeightsHe([1 1 layer.dim 3*layer.dim]));

            layer.q_bias = dlarray(zeros(dim,1));

            layer.k_bias = dlarray(zeros(dim,1));

            layer.v_bias = dlarray(zeros(dim,1));

            layer.proj = dlnetwork(Linear_ ([Name,'_proj'],dim,dim,nc),Initialize = false);

            %%
            % mlp to generate continuous relative position bias

            cpb_mlp = [

            Linear_([Name,'_cpb_mlp_0'],2,512,3)

            reluLayer('Name',[Name,'_cpb_mlp_relu'])

            Linear_([Name,'_cpb_mlp_2'],512,num_head,3)

            ];

            layer.cpb_mlp = dlnetwork(cpb_mlp,Initialize = false);

            layer.relative_position_index = RPI (layer.window_size);  % get pair-wise relative position index for each token inside the window

            layer.relative_coords_table = RCT (layer.window_size,pretrained_window_size); % get relative_coords_table

            %%
            layer.logit_scale = log(10 * ones(1,1,num_head));

            layer.attn_drop = dlnetwork(DropPath([Name,'_attnDropPath'], drop_path_, 1),Initialize = false);

            layer.proj_drop = dlnetwork(DropPath([Name,'_projDropPath'], drop_path_, 1),Initialize = false);

            %%

            % Calculating attention mask for SW-MSA

            if layer.shift_size > 0

                layer.attn_mask = AM (input_resolution,window_size,shift_size);

            else

                layer.attn_mask = [];

            end

        end

        function Z = predict(layer,X) 

            X = reshape(X,layer.input_resolution(2),layer.input_resolution(1),layer.dim,size(X,3));

            X = permute(X,[2,1,3,4]);

            if layer.shift_size > 0 % 判断是否要进行窗口移位

                shifted_x = circshift(X, [-layer.shift_size, -layer.shift_size]);

                X = window_partition(shifted_x, layer.window_size); % 窗口划分

            else

                shifted_x = X;

                X = window_partition(shifted_x, layer.window_size);

            end

            %% 多头QKV

            [N,C,~] = size(X);

            X = reshape(X,layer.window_size,layer.window_size,C,[]);

            X = permute(X,[2,1,3,4]);

            qkvb = cat(1,layer.q_bias,layer.k_bias,layer.v_bias);

            X = dlconv(X, layer.qkvw , qkvb, 'DataFormat', 'SSCB');

            X = permute(X,[2,1,3,4]);

            X = reshape(X,layer.window_size*layer.window_size,size(X,3),[]);

            X = permute(X,[2,1,3]);  % 3*C, WS*WS, B_

            QKV = reshape(X, C/layer.num_head, layer.num_head, 3, layer.window_size*layer.window_size, size(X,3));% C/nh, nh, 3, WS*WS, B_

            QKV = permute(QKV,[3,4,1,2,5]); % 3, WS*WS, C/nh, nh, B_

            q = squeeze(QKV(1,:,:,:,:));

            k = squeeze(QKV(2,:,:,:,:));

            v = squeeze(QKV(3,:,:,:,:));

            %% cosine 相似度计算

            qq = sqrt(sum(q.^2,2));
            q = q ./ qq; % l2 归一化

            kk = sqrt(sum(k.^2,2));
            k = k ./ kk; % l2 归一化

            attn = pagemtimes(q,permute(k,[2,1,3,4,5])); % (N,N,nh,B_)

            ls = exp(min(layer.logit_scale, log(1 / 0.01)));

            attn = attn .* ls;

            %%

            rpi = layer.relative_position_index;

            relative_position_bias_table = layer.cpb_mlp.predict(dlarray(layer.relative_coords_table,'SSCB'));

            relative_position_bias_table = reshape(permute(relative_position_bias_table,[2,1,3,4]),(2*layer.window_size-1)*(2*layer.window_size-1),layer.num_head,[]);

            relative_position_bias = relative_position_bias_table(rpi(:),:);

            relative_position_bias = reshape(relative_position_bias, layer.window_size*layer.window_size, layer.window_size*layer.window_size,[]);% Wh*Ww,Wh*Ww,nH

            relative_position_bias = 16 * sigmoid(relative_position_bias);

            attn = attn + relative_position_bias;

            %%
            if ~isempty(layer.attn_mask)

                nW = size(layer.attn_mask,3);

                mask = reshape(layer.attn_mask,N,N,1,nW);

                attn = reshape(attn,N,N,layer.num_head,[],nW) + reshape(mask,N,N,1,1,[]);

                attn = reshape(attn,N,N,layer.num_head,[]);

                attn = softmax(attn, 'DataFormat', 'UCUU');

            else

                attn = softmax(attn, 'DataFormat', 'UCUU');%(N,N,nh,B_)

            end
            
            %%
            % attn = attn_dropout.predict(attn);

            Z = pagemtimes(attn,v); % (WS*WS, C/NH, NH, B_)

            Z = reshape(Z,size(Z,1), C, size(Z,4)); %(WS*WS, C, B_)

            Z = dlarray(Z,'SSCB');
        
            Z = layer.proj.predict(Z);  % Z = dlconv(Z, layer.qkvw2 , layer.qkvb2, 'DataFormat', 'SSCB');%(WS, WS, C, B_)

            % Z  = proj_drop.predict(Z)

            if  layer.shift_size > 0

                Z = window_reverse(Z,layer.window_size,layer.input_resolution(1),layer.input_resolution(2));

                Z = circshift(Z, [layer.shift_size, layer.shift_size]);

            else

                Z = window_reverse(Z,layer.window_size,layer.input_resolution(1),layer.input_resolution(2));

            end

            Z = permute(Z,[2,1,3,4]);

            Z = reshape(Z,[],size(Z,3),size(Z,4));

        end

        function Z = forward(layer,X)

            X = reshape(X,layer.input_resolution(2),layer.input_resolution(1),layer.dim,size(X,3));

            X = permute(X,[2,1,3,4]);

            if layer.shift_size > 0 % 判断是否要进行窗口移位

                shifted_x = circshift(X, [-layer.shift_size, -layer.shift_size]);

                X = window_partition(shifted_x, layer.window_size); % 窗口划分

            else

                shifted_x = X;

                X = window_partition(shifted_x, layer.window_size);

            end

            %% 多头QKV

            [N,C,~] = size(X);

            X = reshape(X,layer.window_size,layer.window_size,C,[]);

            X = permute(X,[2,1,3,4]);

            qkvb = cat(1,layer.q_bias,layer.k_bias,layer.v_bias);

            X = dlconv(X, layer.qkvw , qkvb, 'DataFormat', 'SSCB');

            X = permute(X,[2,1,3,4]);

            X = reshape(X,layer.window_size*layer.window_size,size(X,3),[]);

            X = permute(X,[2,1,3]);  % 3*C, WS*WS, B_

            QKV = reshape(X, C/layer.num_head, layer.num_head, 3, layer.window_size*layer.window_size, size(X,3));% C/nh, nh, 3, WS*WS, B_

            QKV = permute(QKV,[3,4,1,2,5]); % 3, WS*WS, C/nh, nh, B_

            q = squeeze(QKV(1,:,:,:,:));

            k = squeeze(QKV(2,:,:,:,:));

            v = squeeze(QKV(3,:,:,:,:));

            %% cosine 相似度计算

            qq = sqrt(sum(q.^2,2));
            q = q ./ qq; % l2 归一化

            kk = sqrt(sum(k.^2,2));
            k = k ./ kk; % l2 归一化

            attn = pagemtimes(q,permute(k,[2,1,3,4,5])); % (N,N,nh,B_)

            ls = exp(min(layer.logit_scale, log(1 / 0.01)));

            attn = attn .* ls;

            %%

            rpi = layer.relative_position_index;

            relative_position_bias_table = layer.cpb_mlp.predict(dlarray(layer.relative_coords_table,'SSCB'));

            relative_position_bias_table = reshape(permute(relative_position_bias_table,[2,1,3,4]),(2*layer.window_size-1)*(2*layer.window_size-1),layer.num_head,[]);

            relative_position_bias = relative_position_bias_table(rpi(:),:);

            relative_position_bias = reshape(relative_position_bias, layer.window_size*layer.window_size, layer.window_size*layer.window_size,[]);% Wh*Ww,Wh*Ww,nH

            relative_position_bias = 16 * sigmoid(relative_position_bias);

            attn = attn + relative_position_bias;

            %%
            if ~isempty(layer.attn_mask)

                nW = size(layer.attn_mask,3);

                mask = reshape(layer.attn_mask,N,N,1,nW);

                attn = reshape(attn,N,N,layer.num_head,[],nW) + reshape(mask,N,N,1,1,[]);

                attn = reshape(attn,N,N,layer.num_head,[]);

                attn = softmax(attn, 'DataFormat', 'UCUU');

            else

                attn = softmax(attn, 'DataFormat', 'UCUU');%(N,N,nh,B_)

            end
            
            %%
            attn = dlarray(attn,'SSCB');
            
            attn = layer.attn_drop.forward(attn);
            
            attn = stripdims(attn);

            Z = pagemtimes(attn,v); % (WS*WS, C/NH, NH, B_)

            Z = reshape(Z,size(Z,1), C, size(Z,4)); %(WS*WS, C, B_)

            Z = dlarray(Z,'SSCB');
        
            Z = layer.proj.forward(Z);  % Z = dlconv(Z, layer.qkvw2 , layer.qkvb2, 'DataFormat', 'SSCB');%(WS, WS, C, B_)

            Z  = layer.proj_drop.forward(Z);

            if  layer.shift_size > 0

                Z = window_reverse(Z,layer.window_size,layer.input_resolution(1),layer.input_resolution(2));

                Z = circshift(Z, [layer.shift_size, layer.shift_size]);

            else

                Z = window_reverse(Z,layer.window_size,layer.input_resolution(1),layer.input_resolution(2));

            end

            Z = permute(Z,[2,1,3,4]);

            Z = reshape(Z,[],size(Z,3),size(Z,4));

        end
    end

end