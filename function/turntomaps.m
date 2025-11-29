classdef turntomaps < nnet.layer.Layer

    methods

        function layer = turntomaps (Name)

            arguments

                Name
              
            end

            layer.Name = Name;

            layer.Description = "turntomaps";

        end

        function Z = predict(~,X)

            X = reshape(X,sqrt(size(X,1)),sqrt(size(X,1)),[]);

            Z = permute(X,[2,1,3,4]);
            
        end
    end
end