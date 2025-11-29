classdef PatchMerging < nnet.layer.Layer

    properties      

        input_resolution

        dim %dimension of SuperFeature
           
    end

    methods

        function layer = PatchMerging(Name,input_resolution,dim)

            arguments

               Name

               input_resolution       

               dim   

            end

            layer.input_resolution = input_resolution;

            layer.dim = dim;
            
            layer.Name = Name;

            layer.Description = "PatchMerging";
            
        end


        function Z = predict(layer,X)

            X = reshape(X,layer.input_resolution(2),layer.input_resolution(1),layer.dim,size(X,3));

            X = permute(X,[2,1,3,4]);

            x0 = X(1:2:end, 1:2:end, :, :);  

            x1 = X(2:2:end, 1:2:end, :, :);  

            x2 = X(1:2:end, 2:2:end, :, :);

            x3 = X(2:2:end, 2:2:end, :, :);

            Z = cat(3, x0, x1, x2, x3);  % H/2 W/2 4*C B

            Z = permute(Z,[2,1,3,4]);

            Z = reshape(Z,size(Z,1)*size(Z,2),[],size(Z,4));
                      
        end

    end

end

