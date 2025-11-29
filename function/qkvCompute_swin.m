classdef qkvCompute_swin < nnet.layer.Layer

    properties(Learnable)

        relative_position_bias_table    

        qkv

        proj

        attn_drop

        proj_drop

    end

    properties

        dim

        num_head

        scale

        relative_position_index

        shift_size

        window_size

        input_resolution

        attn_mask

    end

    methods

        function layer = qkvCompute_swin(Name,dim,window_size,num_head,shift_size,input_resolution,nc,drop_path_)

            arguments

                Name

                dim

                window_size

                num_head

                shift_size

                input_resolution

                nc

                drop_path_

            end

            layer.input_resolution = input_resolution;

            layer.dim = dim;

            head_dim = dim/num_head;

            layer.scale = head_dim^(-0.5);

            layer.Name = Name;

            layer.Description = "qkvCompute_swin";

            layer.num_head = num_head;

            layer.window_size = window_size;

            layer.shift_size = shift_size;

            layer.qkv = dlnetwork(Linear_ ([Name,'_qkv'],dim,dim * 3,nc),Initialize = false);

            layer.proj = dlnetwork(Linear_ ([Name,'_proj'],dim,dim, nc),Initialize = false);   

            layer.attn_drop = dlnetwork(DropPath([Name,'_attnDropPath'], drop_path_, 1),Initialize = false);

            layer.proj_drop = dlnetwork(DropPath([Name,'_projDropPath'], drop_path_, 1),Initialize = false);
          
            layer.relative_position_index = RPI (layer.window_size);  

            layer.relative_position_bias_table = zeros((2*window_size-1)*(2*window_size-1),num_head);

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

            X = dlarray(X,'SCB');

            % WS WS C B_ -> WS WS 3*C B_
           
            X = layer.qkv.predict(X); % X = dlconv(X, layer.qkvw1 , layer.qkvb1, 'DataFormat', 'SSCB');

            X = stripdims(X);

            X = permute(X,[2,1,3]); % 3*C, WS*WS, B_

            QKV = reshape(X, C/layer.num_head, layer.num_head, 3, layer.window_size*layer.window_size, size(X,3));% C/nh, nh, 3, WS*WS, B_

            QKV = permute(QKV,[3,4,1,2,5]); % 3, WS*WS, C/nh, nh, B_

            q = squeeze(QKV(1,:,:,:,:));

            k = squeeze(QKV(2,:,:,:,:));

            v = squeeze(QKV(3,:,:,:,:));

            q = q*layer.scale; % Scaling

            %%

            attn = pagemtimes(q,permute(k,[2,1,3,4])); % (WS*WS, WS*WS, NH, B_)

            rpi = layer.relative_position_index;

            relative_position_bias = layer.relative_position_bias_table(rpi(:),:); % 根据相对位置索引取表中bias

            relative_position_bias = reshape(relative_position_bias, layer.window_size*layer.window_size, layer.window_size*layer.window_size, []);% (WS*WS, WS*WS, NH)

            attn = attn + relative_position_bias;

            if ~isempty(layer.attn_mask)

                nW = size(layer.attn_mask,3);

                mask = reshape(layer.attn_mask,N,N,1,nW);

                attn = reshape(attn,N,N,layer.num_head,[],nW) + reshape(mask,N,N,1,1,[]);

                attn = reshape(attn,N,N,layer.num_head,[]);

                attn = softmax(attn, 'DataFormat', 'UCUU');

            else

                attn = softmax(attn, 'DataFormat', 'UCUU');% (WS*WS, WS*WS, NH, B_)

            end

            % attn = layer.attn_drop.predict(attn);

            Z = pagemtimes(attn,v); % (WS*WS, C/NH, NH, B_)

            Z = reshape(Z,size(Z,1), C, size(Z,4)); %(WS*WS, C, B_)

            Z = dlarray(Z,'SCB');
        
            Z = layer.proj.predict(Z);  % Z = dlconv(Z, layer.qkvw2 , layer.qkvb2, 'DataFormat', 'SSCB');%(WS, WS, C, B_)

            % Z  = layer.proj_drop.predict(Z);

            if  layer.shift_size > 0  % 判断是否要进行窗口移位

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

            [N,C,~] = size(X);

            X = dlarray(X,'SCB');

            % WS WS C B_ -> WS WS 3*C B_
           
            X = layer.qkv.forward(X); % X = dlconv(X, layer.qkvw1 , layer.qkvb1, 'DataFormat', 'SSCB');

            X = stripdims(X);

            X = permute(X,[2,1,3]); % 3*C, WS*WS, B_

            QKV = reshape(X, C/layer.num_head, layer.num_head, 3, layer.window_size*layer.window_size, size(X,3));% C/nh, nh, 3, WS*WS, B_

            QKV = permute(QKV,[3,4,1,2,5]); % 3, WS*WS, C/nh, nh, B_

            q = squeeze(QKV(1,:,:,:,:));

            k = squeeze(QKV(2,:,:,:,:));

            v = squeeze(QKV(3,:,:,:,:));

            q = q*layer.scale; % Scaling

            %%

            attn = pagemtimes(q,permute(k,[2,1,3,4])); % (WS*WS, WS*WS, NH, B_)

            rpi = layer.relative_position_index;

            relative_position_bias = layer.relative_position_bias_table(rpi(:),:); % 根据相对位置索引取表中bias

            relative_position_bias = reshape(relative_position_bias, layer.window_size*layer.window_size, layer.window_size*layer.window_size, []);% (WS*WS, WS*WS, NH)

            attn = attn + relative_position_bias;

            if ~isempty(layer.attn_mask)

                nW = size(layer.attn_mask,3);

                mask = reshape(layer.attn_mask,N,N,1,nW);

                attn = reshape(attn,N,N,layer.num_head,[],nW) + reshape(mask,N,N,1,1,[]);

                attn = reshape(attn,N,N,layer.num_head,[]);

                attn = softmax(attn, 'DataFormat', 'UCUU');

            else

                attn = softmax(attn, 'DataFormat', 'UCUU');% (WS*WS, WS*WS, NH, B_)

            end

            attn = dlarray(attn,'SSCB');

            attn = layer.attn_drop.forward(attn);

            attn = stripdims(attn);

            Z = pagemtimes(attn,v); % (WS*WS, C/NH, NH, B_)

            Z = reshape(Z,size(Z,1), C, size(Z,4)); %(WS*WS, C, B_)

            Z = dlarray(Z,'SCB');
        
            Z = layer.proj.forward(Z);  % Z = dlconv(Z, layer.qkvw2 , layer.qkvb2, 'DataFormat', 'SSCB');%(WS, WS, C, B_)

            Z  = layer.proj_drop.forward(Z);

            if  layer.shift_size > 0  % 判断是否要进行窗口移位

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