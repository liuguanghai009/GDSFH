classdef MergeBlock < nnet.layer.Layer

    properties(Learnable)
        
        attn
        
        mlp

        gamma1

        gamma2

        norm1

        norm2


        drop_path

    end

    properties

        dim

        numheads

        mlp_ratio

        hw

        epsilon


    end

    methods

        function layer = MergeBlock(Name,dim,num_heads,nc,epsilon,drop_path_,mlp_ratio,hw)

            arguments

                Name

                dim      

                num_heads

                nc

                epsilon

                drop_path_

                mlp_ratio

                hw

            end

            layer.dim = dim;

            layer.Name = Name;

            layer.numheads = num_heads;

            layer.hw = hw;

            layer.epsilon = epsilon;

            layer.Description = 'MergeBlock';

            layer.norm1 = dlnetwork(Layernorm_ ([Name,'_norm1'],dim,2,epsilon),Initialize = false);

            layer.norm2 = dlnetwork(Layernorm_ ([Name,'_norm2'],dim,2,epsilon),Initialize = false);

            layer.attn =  dlnetwork(Attention([Name,'_attn'],dim,num_heads,nc),Initialize = false);         

            layer.mlp = dlnetwork(MergeFFN(Name,dim,mlp_ratio,hw),Initialize = false);   

            layer.drop_path = dlnetwork(DropPath([Name,'_drop_path'], drop_path_, 1),Initialize = false);

            layer_scale_init_value = 1e-6;

            layer.gamma1 = layer_scale_init_value * ones(1,dim);

            layer.gamma2 = layer_scale_init_value * ones(1,dim);          

        end

        function Z = predict(layer,X)    
            
            X = dlarray(X,'SCB');% (N, C, B) 
            
            X = X + layer.drop_path.predict(layer.gamma1.* dlarray(layer.attn.predict(layer.norm1.predict(X)),'SCB'));

            N = dlarray(size(X,1),"SCB");

            X = dlarray(X,"SCB");

            temp = dlarray(layer.norm2.predict(X),'SCB');

            Z = X + layer.drop_path.predict(layer.gamma2.* layer.mlp.predict(temp));

            Z = stripdims(Z);         
            
        end
        
    end
end