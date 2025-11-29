function  GDSFHexmples

net = importdata('G:/GDSFH/package/ResNet101-epoch50-size448-dim2048.mat');

% im = imread("G:/Example_image/fig1_b.jpg");

im = imread("G:/GDSFH/Example_image/paris_defense_000013.jpg");

%%===================================================================

if size(im,3)==1
    rgb = cat(3,im,im,im);
    im = mat2gray(rgb);
end

img = single(im);

[h, w, ~] = size(img);

rawData = imresize(im, [768 1024]);

if(h < w)
    img_resize = imresize(img, [768 1024]);
else
    img_resize = imresize(img, [1024 768]);
end

 X = activations(net, img_resize, 'res5c_relu', 'OutputAs', 'channels');
 
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

topnum = K/4;
%
for v = 1:topnum
    F(:,:) = F(:,:) + abs(X(:,:,index(v)));%%保持abs()不变
end
%
Hu = F./topnum;

Hu = mat2gray(log(1 + double(Hu)));

%==========================================================================

[hei,wid,~] = size(img_resize);

Fs = imresize(Hu,[hei,wid]);

heatmap_img = ind2rgb(im2uint8(mat2gray(Fs)),parula(256));

imwrite(Fs,['G:/','dam','.jpg']);
imwrite(heatmap_img,['G:/','colordam','.jpg']);
%%=========================================================================
figure;

subplot(1, 3, 1);
imshow(rawData);
%
subplot(1, 3, 2);
imagesc(Fs);
axis off;

%
subplot(1, 3, 3);
imshow(Fs);

axis off;

%%=========================================================================

figure;

subplot(2, 12, 1);

imshow(rawData);
% 
% subplot(2, 12, 2);
% 
% imagesc(Fs);

axis off;

for v = 1:10
    subplot(2, 12, v+2);
    S = imresize(X(:,:,index(v)), [h,w]);
    imagesc(S);
    heatmap_img = ind2rgb(im2uint8(mat2gray(S)),parula(256));
    imwrite(heatmap_img,['G:/',sprintf('%d',v),'.jpg']);
    axis off;
end

for v = 11:20
    subplot(2, 12, v+4);
    S = imresize(X(:,:,index(v)), [h,w]);
    imagesc(S);
    heatmap_img = ind2rgb(im2uint8(mat2gray(S)),parula(256));
    imwrite(heatmap_img,['G:/',sprintf('%d',v),'.jpg']);
    axis off;
end

end