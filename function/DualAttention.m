classdef DualAttention < nnet.layer.Layer

    properties(Learnable)
        q

        kv

        q_proxy

        kv_proxy

        q_proxy_ln

        p_ln

        mlp_proxy

        proxy_ln

        qkv_proxy

        gamma1

        gamma2

        gamma3

        selfatt

        proj

        drop_path

        attn_drop

        proj_drop

    end

    properties

        dim

        numheads

        scale

        epsilon



    end

    methods

        function layer = DualAttention(Name,dim,num_heads,nc,epsilon,drop_path_)

            arguments

                Name

                dim      

                num_heads

                nc

                epsilon

                drop_path_

            end

            layer.dim = dim;

            layer.Name = Name;

            layer.numheads = num_heads;

            head_dim = dim / num_heads;

            layer.scale = head_dim ^(-0.5);

            layer.Description = 'DualAttention';

            layer.NumInputs = 2;

            layer.NumOutputs = 2;

            layer.q = dlnetwork(Linear_ ([Name,'_q'],dim,dim,nc),Initialize = false);

            layer.kv = dlnetwork(Linear_ ([Name,'_kv'],dim,dim *2,nc),Initialize = false);

            layer.proj = dlnetwork(Linear_ ([Name,'_proj'],dim,dim,nc),Initialize = false);

            layer.q_proxy = dlnetwork(Linear_ ([Name,'_q_proxy'],dim,dim,nc),Initialize = false);

            layer.kv_proxy = dlnetwork(Linear_ ([Name,'_kv_proxy'],dim,dim *2,nc),Initialize = false);

            layer.q_proxy_ln = dlnetwork(Layernorm_ ([Name,'_q_proxy_ln'],dim,2,epsilon),Initialize = false);

            layer.p_ln = dlnetwork(Layernorm_ ([Name,'_p_ln'],dim,2,epsilon),Initialize = false);

            layer.drop_path = dlnetwork(DropPath([Name,'_drop_path'], drop_path_, 1),Initialize = false);

            mlp_proxy = [

                Linear_([Name,'_mlp_proxy_0'],dim,dim*4,2)

                reluLayer('Name',[Name,'_mlp_relu'])

                Linear_([Name,'_mlp_proxy_2'],dim*4,dim,2)

                ];

            layer.mlp_proxy = dlnetwork(mlp_proxy,Initialize = false);

            layer.proxy_ln = dlnetwork(Layernorm_ ([Name,'_proxy_ln'],dim,2,epsilon),Initialize = false);


            layer_scale_init_value = 1e-6;

            layer.gamma1 = layer_scale_init_value * ones(1,dim);

            layer.gamma2 = layer_scale_init_value * ones(1,dim);

            layer.gamma3 = layer_scale_init_value * ones(1,dim);
           
            layer.selfatt =  dlnetwork(selfatt(Name,dim,num_heads,nc,epsilon,drop_path_),Initialize = false);         

        end

        function [Z,semantics] = predict(layer,X,semantics)    
            
            X = dlarray(X,'SCB');% (N, C, B) 

            semantics = dlarray(semantics,'SCB');
            semantics_ = layer.selfatt.predict(semantics).*layer.gamma1;
            semantics_ = dlarray(semantics_,'SCB');
            semantics = semantics + layer.drop_path.predict(semantics_); % (N, C, B)            

            [N_p, C_p, B_p] = size(semantics);

            [N, C, B] = size(X);

            X = dlarray(X,'SCB');% (N, C, B) 

            q = layer.q.predict(X); % (N, C, B) -> (N, C, B)

            q = stripdims(q);

            q = reshape(permute(q,[2,1,3]),C/layer.numheads,layer.numheads,N,B); % (C/nh, nh , N, B)

            q = permute(q,[3,1,2,4]); % N, C/nh, nh, B

            q_semantics = layer.q_proxy.predict(layer.q_proxy_ln.predict(semantics)); % (N_p, C_p, B_p)

            q_semantics = stripdims(q_semantics);

            q_semantics = reshape(permute(q_semantics,[2,1,3]),C_p/layer.numheads,layer.numheads,N_p,B_p); % (C_p/nh, nh, N_p, B_p)

            q_semantics = permute(q_semantics,[3,1,2,4]); % N_p, C_p/nh, nh, B_p

            kv_semantics = layer.kv_proxy.predict(X);  % (N, C, B) -> (N, 2C, B)

            kv_semantics = stripdims(kv_semantics);

            kv_semantics = permute(kv_semantics,[2,1,3]);   % 2C N B

            kv_semantics = reshape(kv_semantics, C/layer.numheads, layer.numheads, 2, [], size(X,3));% C/nh, nh, 2, N, B    

            kv_semantics = permute( kv_semantics,[3,4,1,2,5]); % 2, N, C/nh, nh, B

            kp = squeeze(kv_semantics(1,:,:,:,:)); % N, C/nh, nh, B

            vp = squeeze(kv_semantics(2,:,:,:,:)); % N, C/nh, nh, B

            %% Compute the attention weights
            attn = pagemtimes(q_semantics,permute(kp,[2,1,3,4,5])) *layer.scale;  % (N_p, N,  nh, B)

            %% Apply softmax
            attn = softmax(attn, 'DataFormat', 'UCUU');% (N_p, N,  nh, B)

            semantics_ = pagemtimes(attn,vp);% (N_p, C/nh,  nh, B)

            % Merge Heads
            semantics_ = reshape(semantics_, size(semantics_ ,1), [], size(semantics_ ,4)) .*layer.gamma2; %(N_p, C, B)

            semantics_ = dlarray(semantics_,"SCB");

            semantics = semantics + layer.drop_path.predict(semantics_);

            semantics = semantics + layer.drop_path.predict(layer.gamma3.* layer.mlp_proxy.predict(layer.p_ln.predict(semantics))); %(N_p, C, B)

            kv = layer.kv.predict(layer.proxy_ln.predict(semantics));%(N_p, 2C, B)\

            kv = stripdims(kv);

            kv = permute(kv,[2,1,3]);   % 2C N_p B

            kv = reshape(kv, C_p/layer.numheads, layer.numheads, 2, [], size(kv,3));% C/nh, nh, 2, N_p, B    

            kv_semantics = permute(kv,[3,4,1,2,5]); % 2, N_p, C_p/nh, nh, B

            k = squeeze(kv_semantics(1,:,:,:,:)); % N_p, C_p/nh, nh, B

            v = squeeze(kv_semantics(2,:,:,:,:)); % N_p, C_p/nh, nh, B

            %% Compute the attention weights
            attn = pagemtimes(q,permute(k,[2,1,3,4,5])) *layer.scale;  % (N, C/nh, nh, B) (N_p, C_p/nh, nh, B) -> (N, N_p, nh, B)

            %% Apply softmax
            attn = softmax(attn, 'DataFormat', 'UCUU');% (N, N_p, nh, B)

            Z = pagemtimes(attn,v);% (N, N_p, nh, B) (N_p, C_p/nh, nh, B) -> (N, C_p/nh, nh, B)

            Z = reshape(Z,size(Z,1), [], size(Z,4)); %(N, C, B)

            Z = dlarray(Z,"SCB");

            Z = layer.proj.predict(Z); %(N, C, B)

            Z = stripdims(Z);

            semantics = stripdims(semantics);
                       
        end
        
    end
end