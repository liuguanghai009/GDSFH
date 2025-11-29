classdef DWConv < nnet.layer.Layer

    properties

        dim

    end  

    properties(Learnable)

        dwconv

    end

       
    methods

        function layer = DWConv(Name,dim)

            arguments

                Name

                dim

            end

            layer.Name = Name;

            layer.Description = 'DWConv';

            weights = rand(3,3,1,1,dim);
            bias = zeros(1,1,1,dim);

            layer.dwconv = dlnetwork(groupedConvolution2dLayer(3,1,'channel-wise','Name',[Name,'_dwconv_dwconv'],'Padding',1,'Stride',1,'Weights',weights,'Bias',bias),Initialize = false);


        end

        function Z = predict(layer,X)

            X = dlarray(X,"SCB");

            [N,C,B] = size(X);

            X = reshape(X,sqrt(N),sqrt(N),C,B);

            X = permute(X,[2,1,3,4]);

            X = dlarray(X,'SSCB');

            X = layer.dwconv.predict(X);

            X = permute(X,[2,1,3,4]);

            Z = reshape(X,size(X,1)*size(X,2),[],B);
            
            Z = stripdims(Z);
            
        end

    end
    
end