classdef MergeFFN < nnet.layer.Layer

    properties

        dim

        hw


    end  

    properties(Learnable)

        mlp

        fc_proxy

    end

       
    methods

        function layer = MergeFFN(Name,dim,mlp_ratio,hw)

            arguments

                Name

                dim

                mlp_ratio
                
                hw

            end

            layer.Name = Name;

            layer.Description = "MergeFFN";

            layer.hw = hw;

            mlp = [

                Linear_([Name,'_mlp_fc1'],dim,mlp_ratio*dim,2)
    
                DWConv([Name,'_mlp'],mlp_ratio*dim);
    
                geluLayer('Name',[Name,'_mlp_fc_grelu'],'Approximation','tanh')
    
                Linear_([Name,'_mlp_fc2'],mlp_ratio*dim,dim,2)

            ];

            layer.mlp = dlnetwork(mlp,Initialize = false);

            fc_proxy = [

                Linear_([Name,'_mlp_fc_proxy_0'],dim,2*dim,2)
    
                geluLayer('Name',[Name,'_mlp_fc_proxy_grelu'],'Approximation','tanh')
    
                Linear_([Name,'_mlp_fc_proxy_2'],2*dim,dim,2)

            ];

            layer.fc_proxy = dlnetwork(fc_proxy,Initialize = false);

        end

        function Z = predict(layer,X)

            n = layer.hw;

            X = dlarray(X,"SCB");

            x = X(1:n,:,:);

            semantics = X(n+1:end,:,:);

            semantics = layer.fc_proxy.predict(semantics);

            x = layer.mlp.predict(x);

            Z = cat(1,x,semantics);
         
            Z = stripdims(Z);
            
        end

    end
    
end