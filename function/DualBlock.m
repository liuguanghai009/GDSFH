classdef DualBlock < nnet.layer.Layer

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

        epsilon



    end

    methods

        function layer = DualBlock(Name,dim,num_heads,nc,epsilon,drop_path_,mlp_ratio)

            arguments

                Name

                dim      

                num_heads

                nc

                epsilon

                drop_path_

                mlp_ratio

            end

            layer.dim = dim;

            layer.Name = Name;

            layer.numheads = num_heads;

            layer.epsilon = epsilon;

            layer.NumInputs = 2;

            layer.NumOutputs = 2;

            layer.Description = 'DualBlock';

            layer.norm1 = dlnetwork(Layernorm_ ([Name,'_norm1'],dim,2,epsilon),Initialize = false);

            layer.norm2 = dlnetwork(Layernorm_ ([Name,'_norm2'],dim,2,epsilon),Initialize = false);

            layer.attn =  dlnetwork(DualAttention([Name,'_attn'],dim,num_heads,nc,epsilon,drop_path_),Initialize = false);         

             mlp = [

                Linear_([Name,'_mlp_fc1'],dim,mlp_ratio*dim,2)               

                DWConv([Name,'_mlp'],mlp_ratio*dim);

                geluLayer('Name',[Name,'_mlp'],'Approximation','tanh')

                Linear_([Name,'_mlp_fc2'],mlp_ratio*dim,dim,2)

                ];

            layer.mlp = dlnetwork(mlp,Initialize = false);

            layer.drop_path = dlnetwork(DropPath([Name,'_drop_path'], drop_path_, 1),Initialize = false);

            layer_scale_init_value = 1e-6;

            layer.gamma1 = layer_scale_init_value * ones(1,dim);

            layer.gamma2 = layer_scale_init_value * ones(1,dim);          

        end

        function [Z,semantics] = predict(layer,X,semantics)    
            
            X = dlarray(X,'SCB');% (N, C, B) 
            
            X_  = layer.norm1.predict(X);

            X_ = dlarray(X_,'SCB');% (N, C, B) 
            
            semantics = dlarray(semantics,'SCB');

            [X_,semantics] = layer.attn.predict(X_,semantics);

            X_ = dlarray(X_,'SCB');% (N, C, B) 

            Z = X + layer.drop_path.predict(layer.gamma1.* X_);

            Z_ = dlarray(Z,'SCB');% (N, C, B) 

            Z = Z_ + layer.drop_path.predict(dlarray(layer.gamma2.* layer.mlp.predict(dlarray(layer.norm2.predict(Z_),'SCB')),'SCB'));

            Z = stripdims(Z);

            semantics = stripdims(semantics);           
            
        end
        
    end
end