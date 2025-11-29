classdef Layernorm_ < nnet.layer.Layer

    properties

        epsilon

        ndim


    end  

    properties(Learnable)

        Scale

        Offset

    end
       
    methods

        function layer = Layernorm_(Name,dim,ndim,epsilon)

            arguments

                Name

                dim

                ndim

                epsilon

            end

            layer.Name = Name;

            layer.Description = "LayerNorm";

            layer.epsilon = epsilon;

            if ndim == 2

                layer.Scale = ones(1,dim);

                layer.Offset = zeros(1,dim);

            else

                layer.Scale = ones(1,1,dim);

                layer.Offset = zeros(1,1,dim);

            end

            layer.ndim = ndim;

        end

        function Z = predict(layer,X)

            u = mean(X,layer.ndim);

            s = mean((X - u).^2,layer.ndim);

            Z = (X - u)./sqrt(s + layer.epsilon);

            Z = Z.*layer.Scale + layer.Offset;
            
        end

    end
    
end