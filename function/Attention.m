classdef Attention < nnet.layer.Layer

    properties(Learnable)

        q

        kv

        proj


    end

    properties

        dim

        numheads

        scale

        epsilon

    end

    methods

        function layer = Attention(Name,dim,num_heads,nc)

            arguments

                Name

                dim      

                num_heads

                nc

            end

            layer.dim = dim;

            layer.Name = Name;

            layer.numheads = num_heads;

            head_dim = dim / num_heads;

            layer.scale = head_dim ^(-0.5);

            layer.Description = 'Attention';

            layer.q = dlnetwork(Linear_ ([Name,'_q'],dim,dim,nc),Initialize = false);

            layer.kv = dlnetwork(Linear_ ([Name,'_kv'],dim,dim *2,nc),Initialize = false);

            layer.proj = dlnetwork(Linear_ ([Name,'_proj'],dim,dim,nc),Initialize = false);        

        end

        function Z = predict(layer,X)    
            
            X = dlarray(X,'SCB');% (N, C, B)          

            [N, C, B] = size(X);

            q = layer.q.predict(X); % (N, C, B) -> (N, C, B)

            q = stripdims(q);

            q = reshape(permute(q,[2,1,3]),C/layer.numheads,layer.numheads,N,B); % (C/nh, nh , N, B)

            q = permute(q,[3,1,2,4]); % N, C/nh, nh, B         

            kv = layer.kv.predict(X);%(N_p, 2C, B)

            kv = stripdims(kv);

            kv = permute(kv,[2,1,3]);   % 2C N_p B

            kv = reshape(kv, C/layer.numheads, layer.numheads, 2, [], size(kv,3));% C/nh, nh, 2, N_p, B    

            kv = permute(kv,[3,4,1,2,5]); % 2, N_p, C_p/nh, nh, B

            k = squeeze(kv(1,:,:,:,:)); % N_p, C_p/nh, nh, B

            v = squeeze(kv(2,:,:,:,:)); % N_p, C_p/nh, nh, B

            %% Compute the attention weights

            k = stripdims(k);
            q = stripdims(q);
            attn = pagemtimes(q,permute(k,[2,1,3,4,5])) *layer.scale;  % (N, C/nh, nh, B) (N_p, C_p/nh, nh, B) -> (N, N_p, nh, B)

            %% Apply softmax
            attn = softmax(attn, 'DataFormat', 'UCUU');% (N, N_p, nh, B)

            Z = pagemtimes(attn,v) ;% (N, N_p, nh, B) (N_p, C_p/nh, nh, B) -> (N, C_p/nh, nh, B)

            Z = reshape(Z,size(Z,1), [], size(Z,4)); %(N, C, B)

            Z = dlarray(Z,'SSCB');

            Z = layer.proj.predict(Z); %(N, C, B)

            Z = stripdims(Z);
            
        end
        
    end
end