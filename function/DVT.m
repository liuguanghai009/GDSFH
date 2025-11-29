classdef DVT < nnet.layer.Layer

    properties(Learnable)
        block1_0;  block1_1;  block1_2;

        block2_0;  block2_1;  block2_2;  block2_3; block2_4;  block2_5; 

        block3_0;  block3_1;  block3_2;  block3_3; block3_4;  block3_5;
        block3_6;  block3_7;  block3_8;  block3_9; block3_10; block3_11;
        block3_12; block3_13; block3_14; block3_15;block3_16; block3_17;
        block3_18; block3_19; block3_20; block3_21; 
 
        block4_0;  block4_1;  block4_2;

        q       
        q_embed       
        kv
        proxy_ln
        se

        proxy_embed2
        proxy_embed3
        proxy_embed4

        patch_embed1
        patch_embed2
        patch_embed3
        patch_embed4

        norm1
        norm2
        norm3
        norm4

        norm_proxy1
        norm_proxy2
        norm_proxy3 

        post_network

    end

    properties

        embed_dims
        num_classes
        depths
        num_stages
        sep_stage
        scale
        pool
        hw

    end

    methods

        function layer = DVT(Name,in_chans,num_classes,stem_hidden_dim,embed_dims,num_heads,mlp_ratios,drop_path_rate,depths,num_stages,sep_stages,token_label,image_Size)

            arguments

                Name {} = 'dvt'

                in_chans {} = 3

                num_classes {} = 1000

                stem_hidden_dim {} = 32

                embed_dims {} = [64, 128, 320, 448]

                num_heads {} = [2, 4, 10, 14]

                mlp_ratios {} = [8, 8, 4, 3, 2]

                drop_path_rate {} = 0

                depths {} = [3, 4, 6, 3]

                num_stages {} = 4

                sep_stages {} = 2

                token_label {} = 1

                image_Size {} = 384

            end

            layer.Name = Name;
            layer.embed_dims = embed_dims;
            layer.num_classes = num_classes;
            layer.depths = depths;
            layer.num_stages = num_stages;
            layer.sep_stage = sep_stages;
            dpr = linspace(0,drop_path_rate,sum(depths));
            cur = 0;
            nc = 2;

            epsilon1 = 1e-6;
            epsilon2 = 1e-5;

            hw = [image_Size/4 image_Size/8 image_Size/16 image_Size/32].^2;
            
            layer.hw = hw;
            for i = 1:num_stages

                if i == 1
                    patch_embed = Stem (['patch_embed',num2str(i)], in_chans, stem_hidden_dim, embed_dims(i),epsilon2);
                else
                    patch_embed = DownSamples (['patch_embed',num2str(i)], embed_dims(i-1), embed_dims(i),epsilon2);
                end

                if i == 1

                    parminit = squeeze(initializeWeightsHe([1,1,64,embed_dims(i)]));
                    layer.q = parminit;
                    q_embed = [
                        Layernorm_('q_embed_0',embed_dims(i),2,epsilon2)
                        Linear_('q_embed_1',embed_dims(i),embed_dims(i),2)
                        ];
                    layer.q_embed = dlnetwork(q_embed,Initialize = false);
                    layer.pool = dlnetwork(averagePooling2dLayer([7,7],Stride=7,Name='pool'),Initialize = false);
                    layer.kv = dlnetwork(Linear_ ('kv',embed_dims(i),embed_dims(i) *2,nc),Initialize = false);
                    layer.scale = embed_dims(i) ^(-0.5);
                    layer.proxy_ln = dlnetwork(Layernorm_ ('proxy_ln',embed_dims(i),2,epsilon2),Initialize = false);
                    se = [
                        Linear_('se_0',embed_dims(i),embed_dims(i),2)
                        reluLayer('Name','se_relu')
                        Linear_('se_2',embed_dims(i),embed_dims(i)*2,2)
                        ];
                    layer.se = dlnetwork(se,Initialize = false);

                else
                    proj_proxy = [
                        Linear_(['proxy_embed',num2str(i),'_proj_proxy_0'],embed_dims(i-1), embed_dims(i),2)
                        Layernorm_(['proxy_embed',num2str(i),'_proj_proxy_1'],embed_dims(i),2,epsilon2)
                        ];
                    semantic_embed = dlnetwork(proj_proxy,Initialize = false);

                    layer.(['proxy_embed',num2str(i)]) = semantic_embed;

                end

                if i > layer.sep_stage
                    for j = 1:depths(i)
                        if  (mod(j, 2) == 0 && i == 3)
                            mlprat = mlp_ratios(i)-1 ;
                        else
                            mlprat = mlp_ratios(i);
                        end

                        temp = MergeBlock(['block',num2str(i),'_',num2str(j-1)],embed_dims(i),num_heads(i),nc,epsilon1,dpr(cur + j),mlprat,hw(i));

                        layer.(['block',num2str(i),'_',num2str(j-1)]) = dlnetwork(temp,Initialize = false);
                    end

                else
                    for j = 1:depths(i)

                        mlprat = mlp_ratios(i);

                        temp = DualBlock(['block',num2str(i),'_',num2str(j-1)],embed_dims(i),num_heads(i),nc,epsilon1,dpr(cur + j),mlprat);

                        layer.(['block',num2str(i),'_',num2str(j-1)]) = dlnetwork(temp,Initialize = false);
                    end

                end

                if i ~= num_stages

                    norm = dlnetwork(Layernorm_(['norm',num2str(i)],embed_dims(i),2,epsilon1),Initialize = false);

                    layer.(['norm',num2str(i)]) = norm;

                end

                norm_proxy = dlnetwork(Layernorm_(['norm_proxy',num2str(i)],embed_dims(i),2,epsilon1),Initialize = false);

                cur = cur + depths(i);

                layer.(['patch_embed',num2str(i)]) = dlnetwork(patch_embed,Initialize = false);
           

                if i ~= num_stages
                    layer.(['norm_proxy',num2str(i)]) = norm_proxy;
                end

            end

            layer.norm4 = dlnetwork(Layernorm_('norm4',embed_dims(i),2,epsilon1),Initialize = false);

            layer.post_network = dlnetwork(ClassBlock('post_network_0',embed_dims(end),num_heads(end),nc,epsilon1,mlp_ratios(end)),Initialize = false);

        end

        function Z = predict(layer,X)

            %% 多头QKV
            X = dlarray(X,"SSCB");       

            [X,H,W] = layer.patch_embed1.predict(X);

            X = dlarray(X,"SCB");
            H = dlarray(H,"SB");
            W = dlarray(W,"SB");

            [X,semantics] = layer.forward_sep(X,H,W);

            X = dlarray(X,"SSCB");

            semantics = dlarray(semantics,"SCB");

            X = layer.forward_merge(X,semantics);

            X = dlarray(X,"SCB");

            X = layer.forward_last(X);

            X = dlarray(X,"SCB");

            X = layer.norm4.predict(X);

            Z = X;

            Z = stripdims(Z);
        end

        function [Z,semantics] = forward_sep(layer,X,H,W)           

            for i = 1:layer.sep_stage
                

                if i==1

                    X = dlarray(X,'SCB');
                    
                else
                    X = dlarray(X,'SSCB');
                    patch_embed = layer.(['patch_embed',num2str(i)]);
                    [X,H,W] = patch_embed.predict(X);  % (N,C,B)
                end
                
                [~,C,B] = size(X);

                if i == 1
                    X_down = permute(reshape(X,W,H,C,B),[2,1,3,4]); % (H,W,C,B)

                    X_down = dlarray(X_down,'SSCB');

                    X_down = avgpool(X_down,[7,7], "Stride",7);

                    [X_down_H,X_down_W,~,~] = size(X_down);

                    X_down = reshape(permute(X_down,[2,1,3,4]),X_down_H*X_down_W,C,B);

                    X_down = dlarray(X_down,'SCB'); % Nd C B

                    kv = layer.kv.predict(X_down);  % Nd C B -> Nd 2*C B

                    kv = stripdims(kv);

                    kv = permute(kv,[2,1,3]); % 2*C, Nd, B

                    kv = reshape(kv, C, 2, [], size(kv,3));% C, 2, Nd, B

                    kv = permute(kv,[2,3,1,4]); % 2, Nd, C, B

                    k = squeeze(kv(1,:,:,:,:)); % Nd, C, B

                    v = squeeze(kv(2,:,:,:,:)); % Nd, C, B

                    if size(X_down,1) ~= size(layer.q,1)

                        layer_q = permute(reshape(layer.q,8,8,[]),[2,1,3]);

                        layer_q = dlarray(layer_q,"SSB");

                        layer_q = dlresize(layer_q,'OutputSize',[X_down_H,X_down_W],'Method','nearest');

                        layer_q = stripdims(layer_q);

                        layer_q = permute(layer_q,[2,1,3]);

                        layer_q = reshape(layer_q,[],size(layer_q,3)); % Nd, C

                    else

                        layer_q =layer.q; % Nd, C

                    end

                    layer_q = dlarray(layer_q,'SCB');

                    layer_q = layer.q_embed.predict(layer_q); % Nd, C

                    layer_q = stripdims(layer_q);

                    attn = pagemtimes(layer_q,permute(k,[2,1,3]))*layer.scale; % (Nd, Nd, B)

                    attn = softmax(attn, 'DataFormat', 'UCUU');

                    semantics = pagemtimes(attn,v); % (Nd, C, B)

                    semantics = permute(semantics,[1,2,4,3]);

                    X_down = stripdims(X_down);

                    semantics = cat(3,semantics,permute(X_down,[1,2,4,3])); % (Nd, C, 2, B)

                    se = mean(squeeze(sum(semantics,3)),1);

                    se = dlarray(se,'SCB');

                    se = layer.se.predict(se);

                    se = reshape(se,C, 2,[]); % (C, 2, B)

                    se = softmax(se, 'DataFormat', 'UCU');

                    se = permute(se,[4,1,2,3]); % (1, C, 2, B)

                    semantics = squeeze(sum(se.*semantics,3)); % (Nd, C, 2, B)

                    semantics = dlarray(semantics,'SCB');

                    semantics = layer.proxy_ln.predict(semantics); % (WS*WS, C, B)

                else

                    semantics_embed = layer.(['proxy_embed',num2str(i)]);
                     
                    semantics = semantics_embed.predict(semantics);

                end

                for blk = 1:layer.depths(i)
                    block = layer.(['block',num2str(i),'_',num2str(blk-1)]);
                    X = dlarray(X,'SCB');
                    semantics = dlarray(semantics,'SCB');
                    [X,semantics] = block.predict(X,semantics);
                end

                norm = layer.(['norm',num2str(i)]);
                X = norm.predict(X);
                
                X = reshape(X,sqrt(size(X,1)),sqrt(size(X,1)),C,B);
                X = permute(X,[2,1,3,4]);
                norm_semantics = layer.(['norm_proxy',num2str(i)]);
                semantics = dlarray(semantics,'SCB');
                semantics = norm_semantics.predict(semantics);

            end

            Z = stripdims(X);

            semantics = stripdims(semantics);

        end

        function Z = forward_merge(layer,X,semantics)

            for i = layer.sep_stage+1:layer.num_stages
                patch_embed = layer.(['patch_embed',num2str(i)]);

                X = dlarray(X,'SSCB');
                [X,H,W] = patch_embed.predict(X);
                semantics_embed= layer.(['proxy_embed',num2str(i)]);
                semantics = dlarray(semantics,'SCB');
                semantics = semantics_embed.predict(semantics);
                X = stripdims(X);
                X = cat(1,X,semantics);
                for blk = 1:layer.depths(i)
                    block = layer.(['block',num2str(i),'_',num2str(blk-1)]);
                    X = dlarray(X,'SCB');
                    X = block.predict(X);
                end
                lasthw = layer.hw(i);
                semantics = X(lasthw+1:end,:,:);
                X = X(1:lasthw,:,:);
                B = size(X,3);
                if i ~= layer.num_stages
                    norm = layer.(['norm',num2str(i)]);
                    X = dlarray(X,'SCB');
                    X = norm.predict(X);
                    X = reshape(X,sqrt(lasthw),sqrt(lasthw),[],B);
                    X = permute(X,[2,1,3,4]);
                    norm_semantics = layer.(['norm_proxy',num2str(i)]);
                    semantics = dlarray(semantics,'SCB');
                    semantics = norm_semantics.predict(semantics);
                end
            end

            Z = cat(1,X,semantics);
            Z = stripdims(Z);

        end

        function Z = forward_last(layer,X)

            X = dlarray(X,"SCB");

            cls_tokens = mean(X,1);

            Z = cat(1,cls_tokens,X);

            Z = dlarray(Z,'SCB');

            Z = layer.post_network.predict(Z);

            Z = stripdims(Z);

        end

    end
end