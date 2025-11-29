classdef FFN < nnet.layer.Layer

    properties

        dim

    end  

    properties(Learnable)

        mlp

    end

       
    methods

        function layer = FFN(Name,dim,mlp_ratio)

            arguments

                Name

                dim

                mlp_ratio
               

            end

            layer.Name = Name;

            layer.Description = "FFN";


            mlp = [

                Linear_([Name,'_mlp_fc1'],dim,mlp_ratio*dim,2)
    
                geluLayer('Name',[Name,'_mlp_fc_grelu'],'Approximation','tanh')
    
                Linear_([Name,'_mlp_fc2'],mlp_ratio*dim,dim,2)

            ];

            layer.mlp = dlnetwork(mlp,Initialize = false);

        end

        function Z = predict(layer,X)

            X = dlarray(X,"SCB");

            X = layer.mlp.predict(X);
    
            Z = stripdims(X);
            
        end

    end
    
end