classdef GeMpool < nnet.layer.Layer
    properties(Learnable)
        p
    end
    properties
        epsilon

    end

    methods
        function layer = GeMpool(Name)
            arguments
                Name
            end
            layer.Name = Name;
            layer.Description = "GeMpool";
            layer.epsilon = 1e-6;
            layer.p = 3;
            layer = layer.setLearnRateFactor('p',10);
        end

        function Z=predict(layer,X)

            Z = max(sum(X.^layer.p,[1,2]).^(1/layer.p),layer.epsilon);

        end

    end
end