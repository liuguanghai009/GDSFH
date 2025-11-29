function [featureMap,clsToken] = RerangeToFeaturemaps(features)

if mod(size(features,1),2) == 0 
    %空间数量为偶数，则不包含分类向量clsToken
    hw = sqrt(size(features,1));
    featureMap = reshape(features,hw,hw,[]);
    featureMap = permute(featureMap,[2,1,3]);
    clsToken = [];

else
    %空间数量为奇数，则包含分类向量clsToken 
    clsToken = features(1,:);
    features(1,:) = [];
    hw = sqrt(size(features,1));
    featureMap = reshape(features,hw,hw,[]);
    featureMap = permute(featureMap,[2,1,3]);
   
end
 

end