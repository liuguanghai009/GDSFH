classdef ClassBlock < nnet.layer.Layer

    properties(Learnable)
        
        attn
        
        mlp

        norm1

        norm2

    end

    properties

        dim

        numheads

        mlp_ratio

        epsilon
    end

    methods

        function layer = ClassBlock(Name,dim,num_heads,nc,epsilon,mlp_ratio)

            arguments

                Name

                dim      

                num_heads

                nc

                epsilon

                mlp_ratio

            end

            layer.dim = dim;

            layer.Name = Name;

            layer.numheads = num_heads;

            layer.Description = 'ClassBlock';

            layer.epsilon = epsilon;

            layer.norm1 = dlnetwork(Layernorm_ ([Name,'_norm1'],dim,2,epsilon),Initialize = false);

            layer.norm2 = dlnetwork(Layernorm_ ([Name,'_norm2'],dim,2,epsilon),Initialize = false);

            layer.attn =  dlnetwork(ClassAttention([Name,'_attn'],dim,num_heads,nc),Initialize = false);         

            layer.mlp = dlnetwork(FFN(Name,dim,mlp_ratio),Initialize = false);       

        end

        function Z = predict(layer,X)    
            
            X = dlarray(X,'SCB');% (N, C, B) 

            cls_embed = X(1,:,:);

            cls_embed = dlarray(cls_embed,"SCB");
            
            cls_embed = cls_embed +layer.attn.predict(dlarray(layer.norm1.predict(X),'SCB'));

            cls_embed = dlarray(cls_embed,"SCB");

            cls_embed_ = dlarray(layer.norm2.predict(cls_embed),'SCB');

            cls_embed = cls_embed + layer.mlp.predict(cls_embed_);

            Z = cat(1,cls_embed,X(2:end,:,:));

            Z = stripdims(Z);         
            
        end
        
    end
end