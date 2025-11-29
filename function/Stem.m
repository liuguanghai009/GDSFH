classdef Stem < nnet.layer.Layer

    properties

        in_channels

        hidden_dim

        out_channels

        epsilon

    end

    properties(Learnable)

        norm

    end
    
    methods
        
        function layer = Stem(Name,in_channels,stem_hidden_dim,out_channels,epsilon)

            arguments

               Name

               in_channels

               stem_hidden_dim

               out_channels

               epsilon

            end

            layer.Name = Name;

            layer. in_channels = in_channels;

            layer.hidden_dim = stem_hidden_dim;

            layer.out_channels = out_channels;

            layer.Description = 'Stem';

            layer.NumOutputs = 3;

            layer.norm = dlnetwork(Layernorm_([Name,'_norm'],out_channels,2,epsilon),Initialize = false);
 
        end

        function [Z,H,W] = predict(layer,X)

            X = dlarray(X,"SSCB");

            [H,W,C,B] = size(X);

            X = permute(X,[2,1,3,4]);

            X = reshape(X,H*W,C,B);

            X = dlarray(X,"SCB");

            H = dlarray(H);

            W = dlarray(W);

            Z = layer.norm.predict(X); % (N,C,B)

            Z = stripdims(Z);

        end

    end
end