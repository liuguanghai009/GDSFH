classdef PatchEmbed < nnet.layer.Layer

    properties(Learnable)

        Weights

        Bias

    end

    properties

        img_size

        patch_size

        patch_resolution

        num_patches

        in_chans

        embed_dim

        isreshape
        
    end
    
    methods
        
        function layer = PatchEmbed(Name,img_size,patch_size,in_chans,embed_dim,isreshape)

            arguments

               Name

               img_size

               patch_size

               in_chans

               embed_dim

               isreshape

            end

            patch_size = [patch_size,patch_size];

            patch_resolution = [img_size(1)/patch_size(1),img_size(2)/patch_size(2)];

            layer.Name = Name;

            layer.Description = "PatchEmbed";

            layer.img_size = img_size;

            layer.patch_size = patch_size;

            layer.patch_resolution = patch_resolution;

            layer.num_patches = patch_resolution(1)*patch_resolution(2);

            layer.in_chans= in_chans;

            layer.embed_dim = embed_dim;

            layer.Weights = initializeWeightsHe([patch_size(1) patch_size(2) in_chans embed_dim]);
            
            layer.Bias = zeros(embed_dim,1);

            layer.isreshape = isreshape;

        end

        function Z = predict(layer,X)

            Z = dlconv(X, layer.Weights, layer.Bias, 'DataFormat', 'SSCB','Stride',layer.patch_size);

            [H,W,C,B] = size(Z);
            
            if layer.isreshape == 1

                Z = permute(Z,[2,1,3,4]);
    
                Z = reshape(Z,H*W,C,B);

            end

        end

    end
end