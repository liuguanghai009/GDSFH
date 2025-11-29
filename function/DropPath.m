classdef  DropPath < nnet.layer.Layer & nnet.layer.Formattable% (Optional)

    properties
        % (Optional) Layer properties.

        % Layer properties go here.
        drop_prob

        scale_by_keep

    end
    
    methods
        function layer =  DropPath(Name, drop_prob, scale_by_keep)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.
            arguments
                Name
                drop_prob 
                scale_by_keep 
            end

            layer.drop_prob = drop_prob;

            layer.scale_by_keep = scale_by_keep;

            layer.Name = Name;

            % Layer constructor function goes here.
        end
        
        function Z = predict(~, X)
            % Forward input data through the layer at prediction time and
            % output the result.

            Z = X;

        end

        function Z = forward(layer, X)
            % Forward input data through the layer at prediction time and
            % output the result.

            if layer.drop_prob == 0

                Z = X;

            else

                keep_prob = 1 - layer.drop_prob;

                if ndims(X) <= 3

                    nd = 4;

                else

                    nd = ndims(X);

                end

                shape = [ones(1, nd - 1), size(X, nd)];

                random_tensor = keep_prob + rand(shape);

                random_tensor = floor(random_tensor);

                if keep_prob > 0 && layer.scale_by_keep

                    X = X / keep_prob;

                end

                Z = X .* random_tensor;

            end

        end
    end
end