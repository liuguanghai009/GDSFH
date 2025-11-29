classdef LayerScale < nnet.layer.Layer

    properties

        embed_dim

    end

    properties(Learnable)

        weight

    end
       
    methods

        function layer = LayerScale (Name,embed_dim,ifls)

            arguments

                Name
                
                embed_dim

                ifls

            end

            layer.Name = Name;

            layer.Description = "LayerScale";

            layer.embed_dim = embed_dim;

            if ifls == 1
                
                layer.weight = 1e-5 *ones(1,embed_dim);

            else

                layer.weight = ones(1,embed_dim);

                layer = layer.setLearnRateFactor('weight',0);

            end

        end

        function Z = predict(layer,X)

            Z = X.*layer.weight;

             end
            
        end
end