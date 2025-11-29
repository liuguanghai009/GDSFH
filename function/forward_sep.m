classdef forward_sep < nnet.layer.Layer

    properties(Learnable)

    end

    properties

        embed_dims
    end

    methods

        function layer = forward_sep(Name,embed_dims)

            arguments

                Name
 
                embed_dims

            end

            layer.Name = Name;

            layer.pool = dlnetwork(averagePooling2dLayer([7,7],Stride=7,Name='pool'),Initialize = false);

            layer.kv = dlnetwork(Linear_ ('kv',embed_dims,embed_dims *2,nc),Initialize = false);

            layer.scale = embed_dims ^(-0.5);

            layer.proxy_ln = dlnetwork(Layernorm_ ('proxy_ln',embed_dims,2,epsilon),Initialize = false);

            se = [
                Linear_('se_0',embed_dims,embed_dims,2)

                reluLayer('Name','se_relu')

                Linear_('se_2',embed_dims,embed_dims*2,2)

                ];

            layer.se = dlnetwork(se,Initialize = false);

            parminit = squeeze(initializeWeightsHe([1,1,64,embed_dims]));

            layer.q = parminit;

            q_embed = [

                Layernorm_('q_embed_1',embed_dims,2,epsilon)

                Linear_('q_embed_2',embed_dims,embed_dims,2)
            
            ];

            layer.q_embed = dlnetwork(q_embed,Initialize = false);

            layer.selfatt =  dlnetwork(qkvCompute_vit(Name,dim,num_head,nc,drop_path_),Initialize = false);

        end

        function [Z,semantics] = predict(layer,X)         

             %% 多头QKV
            [N,C,B] = size(X);

            X = dlarray(X,'SCB');

            X = permute(reshape(X,sqrt(N),sqrt(N),C,B),[2,1,3,4]);

            X_down = layer.pool.predict(X);

            [X_down_H,X_down_W,~,~] = size(X_down);

            X_down = reshape(permute(X_down,[2,1,3,4]),N,C,B);

            % WS*WS C B -> WS*WS 2*C B
          
            kv = layer.kv.predict(X_down); 

            kv = stripdims(kv);

            kv = permute(kv,[2,1,3]); % 2*C, WS*WS, B

            kv = reshape(kv, C, 2, layer.window_size*layer.window_size, size(X,3));% C, 2, WS*WS, B

            kv = permute(kv,[2,3,1,4]); % 2, WS*WS, C, B

            k = squeeze(kv(2,:,:,:,:)); % WS*WS, C, B

            v = squeeze(kv(3,:,:,:,:)); % WS*WS, C, B

            if size(X_down,2) ~= size(layer.q,1)

                layer_q = permute(reshape(layer.q,8,8,-1),[2,1,3]);

                layer_q = dlresize(layer_q,'OutputSize',[X_down_H,X_down_W],'Method','linear');

                layer_q = reshape(layer_q,-1,size(layer.q,3)); % WS*WS, C

            end
            
            layer_q = dlarray(layer_q,'SCB');

            layer_q = layer.q_embed.predict(layer_q); % WS*WS, C

            attn = pagemtimes(layer_q,permute(k,[2,1,3])); % (WS*WS, WS*WS, B)

            attn = softmax(attn, 'DataFormat', 'UCUU');

            semantics = pagemtimes(attn,v); % (WS*WS, C, B)

            semantics = permute(dlarray(semantics,'SCB'),[1,2,4,3]); 

            semantics = cat(3,semantics,permute(X_down,[1,2,4,3])); % (WS*WS, C, 2, B)

            se = mean(squeeze(sum(semantics,3)),1);

            se = layer.se.predict(se);

            se = reshape(se,C, 2,-1); % (C, 2, B)

            se = softmax(se, 'DataFormat', 'CUU');

            se = permute(se,[4,1,2,3]); % (1, C, 2, B)

            se = squeeze(sum(se.*semantics,3)); % (WS*WS, C, B)
        
            semantics = layer.proxy_ln.predict(se); % (WS*WS, C, B)

            Z = stripdims(Z);

            semantics = stripdims(semantics);
           
     
        end

    end

end