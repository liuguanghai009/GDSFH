classdef selfatt < nnet.layer.Layer

    properties(Learnable)

        qkv_proxy

        proj


    end

    properties

        dim

        num_head

        scale


    end

    methods

        function layer = selfatt(Name,dim,num_head,nc,epsilon,drop_path_)

            arguments

                Name

                dim      

                num_head

                nc

                epsilon

                drop_path_

            end

            layer.dim = dim;

            head_dim = dim/num_head;

            layer.scale = head_dim^(-0.5);

            layer.Name = Name;

            layer.Description = 'selfatt';

            layer.num_head = num_head;

            qkv_proxy = [

                Layernorm_([Name,'_qkv_proxy_0'],dim,2,epsilon)
                
                Linear_([Name,'_qkv_proxy_1'],dim,dim*3,2)         
            
            ];
           
            layer.qkv_proxy = dlnetwork(qkv_proxy,Initialize = false);

        end

        function Z = predict(layer,X)    

            X = dlarray(X,'SCB');

            [N,C,~] = size(X);

            X = layer.qkv_proxy.predict(X); % (WS, WS, C, B_) -> (WS, WS, 3*C, B_)

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
            
        end
        
       
    end
end