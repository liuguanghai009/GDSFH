function [arr1 ,arr2] = Hu_based_representation(X,Y,im)
%
[hei,wid,K] = size(X);
[h,w,~] = size(Y);
%
XF = zeros(hei,wid,K);
YF = zeros(h,w,K);

F = original_Hu_feature(X,im,(K/4)); %%维持k/4不变

for i=1:K

    [~, ~, Gv, Gh] = edge(X(:,:,i),'Sobel');

    X(:,:,i) = sqrt(Gv.*Gv + Gh.*Gh);
    %
    [~, ~, Gv, Gh] = edge(X(:,:,i),'Sobel');

    X(:,:,i)  = sqrt(Gv.*Gv + Gh.*Gh);
    %
    XF(:,:,i) =  X(:,:,i).*F;

    %%===================================%%

    [~, ~, Gv, Gh] = edge(Y(:,:,i),'Sobel');

    Y(:,:,i) = sqrt(Gv.*Gv + Gh.*Gh);

    %
    [~, ~, Gv, Gh] = edge(Y(:,:,i),'Sobel');

    Y(:,:,i) = sqrt(Gv.*Gv + Gh.*Gh);
    %

    YF(:,:,i) = Y(:,:,i);
    %
end

SWX = sum(XF,[1,2]);
SWY = sum(YF,[1,2]);
%
arr1 = zeros(1,K);
arr2 = zeros(1,K);

for i=1:K
    arr1(1,i) = SWX(1,i);
    arr2(1,i) = SWY(1,i);
end
%
end