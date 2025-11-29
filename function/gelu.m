classdef gelu < nnet.layer.Layer
    
    methods

        function layer=gelu(Name)

            arguments

                Name

            end

            layer.Name=Name;

            layer.Description="gelu";
                             
        end
        
        function Z=predict(~,X)
            
            Z = 0.5 * X .* (1 + tanh(sqrt(2/pi) * (X + 0.044715 * X.^3)));

        end

    end
    
end