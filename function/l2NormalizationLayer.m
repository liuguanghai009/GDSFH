classdef l2NormalizationLayer < nnet.layer.Layer
    
    properties
         epsilon
    end
    
    methods
        function layer=l2NormalizationLayer(Name)
            arguments
               Name
            end
            layer.Name = Name;
            layer.Description = "l2";
            layer.epsilon = 1e-6 ;
        end
        
        function Z = predict(layer,X)
             
             massp = sum(X.^2) + layer.epsilon;

             massp = massp.^(1/2);

             Z = X./massp;

        end
    end
end
