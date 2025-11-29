classdef format_output < nnet.layer.Layer

    properties

        out_type

    end
       
    methods

        function layer = format_output (Name, out_type)

            arguments

                Name

                out_type
              
            end

            layer.Name = Name;

            layer.out_type = out_type;

            layer.Description = "format_output";

        end

        function Z = predict(layer,X)

            switch layer.out_type

                case 'cls_token'

                    Z = X(1,:,:,:);

                case 'avgp'

                    Z = mean(X,1);

            end

            Z = permute(Z,[1,4,2,3]);
            
        end
    end
end