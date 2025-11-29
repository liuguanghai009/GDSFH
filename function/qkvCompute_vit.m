classdef qkvCompute_vit < nnet.layer.Layer

    properties(Learnable)

        qkv

        proj

        attn_drop

        proj_drop

    end

    properties

        dim

        num_head

        scale


    end

    methods

        function layer = qkvCompute_vit(Name,dim,num_head,nc,drop_path_)

            arguments

                Name

                dim      

                num_head

                nc

                drop_path_

            end


            layer.dim = dim;

            head_dim = dim/num_head;

            layer.scale = head_dim^(-0.5);

            layer.Name = Name;

            layer.Description = "qkvCompute_vit";

            layer.num_head = num_head;

            layer.qkv = dlnetwork(Linear_ ([Name,'_qkv'],dim,dim * 3,nc),Initialize = false);

            layer.proj = dlnetwork(Linear_ ([Name,'_proj'],dim,dim,nc),Initialize = false); 

            layer.attn_drop = dlnetwork(DropPath([Name,'_attnDropPath'], drop_path_, 1),Initialize = false);

            layer.proj_drop = dlnetwork(DropPath([Name,'_projDropPath'], drop_path_, 1),Initialize = false);

        end

        function Z = predict(layer,X)    

            X = dlarray(X,'SCB');

            [N,C,~] = size(X);

            X = layer.qkv.predict(X); % (WS, WS, C, B_) -> (WS, WS, 3*C, B_)

            X = stripdims(X);

            X = permute(X,[2,1,3]);   % 3*C WS*WS B_

            QKV = reshape(X, C/layer.num_head, layer.num_head, 3, [], size(X,3));% C/nh, nh, 3, WS*WS, B_

            QKV = permute(QKV,[3,4,1,2,5]); % 3, WS*WS, C/nh, nh, B_

            q = squeeze(QKV(1,:,:,:,:));

            k = squeeze(QKV(2,:,:,:,:));

            v = squeeze(QKV(3,:,:,:,:));

            %% Compute the attention weights
            attn = pagemtimes(q,permute(k,[2,1,3,4,5])) *layer.scale;  % (WS*WS, WS*WS, NH, B_)

            %% Apply softmax
            attn = softmax(attn, 'DataFormat', 'UCUU');% (WS*WS, WS*WS, NH, B_)

            %% Apply dropout
            % attn = attn_dropout.predict(attn);

            %% Compute the attention
            Z = pagemtimes(attn,v); % (WS*WS, C/NH, NH, B_)

            % Merge Heads
            Z = reshape(Z,size(Z,1), [], size(Z,4)); %(WS*WS, C, B_)

            %%
            Z = dlarray(Z,'SCB');
        
            Z = layer.proj.predict(Z);  % Z = dlconv(Z, layer.qkvw2 , layer.qkvb2, 'DataFormat', 'SSCB');%(WS, WS, C, B_)

            % Z  = proj_drop.predict(Z) 

            Z = stripdims(Z);
            
        end
        
        function Z = forward(layer,X)

            X = dlarray(X,'SCB');

            [N,C,~] = size(X);

            X = layer.qkv.predict(X); % (WS, WS, C, B_) -> (WS, WS, 3*C, B_)

            X = stripdims(X);

            X = permute(X,[2,1,3]);   % 3*C WS*WS B_

            QKV = reshape(X, C/layer.num_head, layer.num_head, 3, [], size(X,3));% C/nh, nh, 3, WS*WS, B_

            QKV = permute(QKV,[3,4,1,2,5]); % 3, WS*WS, C/nh, nh, B_

            q = squeeze(QKV(1,:,:,:,:));

            k = squeeze(QKV(2,:,:,:,:));

            v = squeeze(QKV(3,:,:,:,:));

            %% Compute the attention weights
            attn = pagemtimes(q,permute(k,[2,1,3,4,5])) *layer.scale;  % (WS*WS, WS*WS, NH, B_)

            %% Apply softmax
            attn = softmax(attn, 'DataFormat', 'UCUU');% (WS*WS, WS*WS, NH, B_)

            %% Apply dropout

            attn = dlarray(attn,'SSCB');

            attn = layer.attn_drop.forward(attn);

            attn = stripdims(attn);

            %% Compute the attention
            Z = pagemtimes(attn,v); % (WS*WS, C/NH, NH, B_)

            % Merge Heads
            Z = reshape(Z,size(Z,1), [], size(Z,4)); %(WS*WS, C, B_)

            %%
            Z = dlarray(Z,'SCB');
        
            Z = layer.proj.predict(Z);  % Z = dlconv(Z, layer.qkvw2 , layer.qkvb2, 'DataFormat', 'SSCB');%(WS, WS, C, B_)

            Z  = layer.proj_drop.forward(Z); 

            Z = stripdims(Z);
            
        end
    end
end