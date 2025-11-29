function Hu = original_Hu_feature(X,im,topnum)

[hei,wid,K] = size(X);

rankw = zeros(1,K);

im = rgb2gray(im);

%%按照论文内容，原始输入图像不用缩放，UBK图库保持原来图像大小

imdata = log(1 + double(im));%%保持log()不变

phi0 = inmoments(imdata);
%
F = zeros(hei,wid);
%
for i=1:K
    % 
    X(:,:,i) = log(1 + double(X(:,:,i))); %%保持log()不变
    % 
    phi1 = inmoments(X(:,:,i));
    %
    for j=1:7
        rankw(1,i) = rankw(1,i) + abs(phi0(j) - phi1(j))/(abs(phi0(j)) + abs(phi1(j)));
    end
    %
end
%
[~,index] = sort(rankw,'ascend');
%
for v = 1:topnum
    F(:,:) = F(:,:) + abs(X(:,:,index(v)));%%保持abs()不变
end
%
 Hu = F./topnum;
%
Hu = mat2gray(log(1 + double(Hu)));
%
end