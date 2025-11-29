classdef Linear_ < nnet.layer.Layer


    properties

        id

        od

        nc

    end

    properties(Learnable)

        Weights

        Bias

    end
       
    methods

        function layer = Linear_ (Name,indim,outdim,nc)

            arguments

                Name
                
                indim
                
                outdim

                nc

            end

            layer.Name = Name;

            layer.Description = "Linear_layer";

            layer.Weights = randn([indim, outdim]) * sqrt(2 / (outdim) );

            layer.Bias = zeros(1,outdim);

            layer.od = outdim;

            layer.id = indim;

            layer.nc = nc;

        end

        function Z = predict(layer,X)

            if layer.nc == 3

                [H,W,C,B] = size(X);

                X = permute(X,[2,1,3,4]);

                X = reshape(X,H*W,C,B);

            end

            Z = pagemtimes(X,layer.Weights) + layer.Bias;

            if layer.nc == 3

                Z = reshape(Z,H,W,layer.od,B);

                Z = permute(Z,[2,1,3,4]);

            end
            
        end

    end
end