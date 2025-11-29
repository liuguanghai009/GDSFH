classdef pos_embed_ < nnet.layer.Layer

    properties

        image_size

        patch_size

        embed_dim

        num_prefix_tokens

    end

    properties(Learnable)

        cls_token

        pos_embed

    end

    methods

        function layer = pos_embed_ (Name, embed_dim, image_size, patch_size)

            arguments

                Name

                embed_dim

                image_size

                patch_size

            end

            layer.Name = Name;

            layer.Description = "pos_embed";

            layer.num_prefix_tokens = 1;

            layer.embed_dim = embed_dim;

            layer.image_size = image_size;

            layer.patch_size = patch_size;

            num_patches = (image_size(1) / patch_size) * (image_size(2) / patch_size);

            layer.cls_token = zeros(layer.num_prefix_tokens, embed_dim, 1);

            embed_len = num_patches + layer.num_prefix_tokens;

            layer.pos_embed = rand(embed_len, embed_dim, 1) * .02;

        end

        function Z = predict(layer,X)

            [H,W,C,B] = size(X);

            src_shape(1) = layer.image_size(1) / layer.patch_size;

            src_shape(2) = layer.image_size(2) / layer.patch_size;

            if src_shape(1) ~= H || src_shape(2) ~= W % 如果输入尺寸不等于模型预训练尺寸，则需要reszie pos_embed参数

                extra_tokens = layer.pos_embed(layer.num_prefix_tokens,:,:); % 先取出 tokens 

                src_weight = layer.pos_embed(layer.num_prefix_tokens + 1:end,:,:);

                src_weight = reshape(src_weight,src_shape(1),src_shape(2),layer.embed_dim,[]);

                src_weight = permute(src_weight,[2,1,3,4]);

                src_weight = dlarray(src_weight,'SSCB');

                dst_weight = dlresize(src_weight,'OutputSize',[H,W]); % 默认是 ' nearest '

                dst_weight = stripdims(dst_weight);

                dst_weight = permute(dst_weight,[2,1,3,4]);

                dst_weight = reshape(dst_weight,H*W,size(dst_weight,3),size(dst_weight,4));

                dst_weight = cat(1,extra_tokens,dst_weight);

            else

                dst_weight = layer.pos_embed;

            end

            X = permute(X,[2,1,3,4]);

            X = reshape(X,H*W,C,B);

            X = cat(1,repmat(layer.cls_token,1,1,B),X);

            Z = X + dst_weight;

        end

    end
end