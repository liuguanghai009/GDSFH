classdef DownSamples < nnet.layer.Layer

    properties

        dim

    end  

    properties(Learnable)

        proj
        norm

    end

       
    methods

        function layer = DownSamples(Name,inc,dim,epsilon)

            arguments

                Name

                inc

                dim

                epsilon

            end

            layer.Name = Name;

            layer.dim = dim;

            layer.Description = 'DownSamples';

            weights = rand(3,3,inc,dim);
            
            bias = zeros(1,1,dim);

            layer.NumOutputs = 3;

            layer.proj = dlnetwork(convolution2dLayer(3,dim,'Stride',2,'Padding',1,'Name',[Name,'_proj'],'Weights',weights,'Bias',bias),Initialize = false);

            layer.norm = dlnetwork(Layernorm_ ([Name,'_norm'],dim,2,epsilon),Initialize = false);

        end

        function [Z,H,W] = predict(layer,X)

            X = dlarray(X,'SSCB');

            [H,W,C,B] = size(X);

            X = layer.proj.predict(X);

            X = reshape(permute(X,[2,1,3,4]),[],layer.dim,B);

            H = dlarray(H);
            W = dlarray(W);

            X = dlarray(X,"SSCB");
            Z = layer.norm.predict(X);

            Z = stripdims(Z);
            
        end

    end
    
end