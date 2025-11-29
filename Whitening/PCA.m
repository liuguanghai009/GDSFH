function X = PCA(Y,dim)

 [~,scoreTrain,~,~,~,~] = pca(Y);

X = scoreTrain(:,1:dim);

end