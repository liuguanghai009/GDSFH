function X = whitening_V2(XTrain,XTest,dim)

Y = transfer_learning_whitening(XTrain,XTest,dim); %%负例用来训练参数，供给给正例用

X = PCA([Y XTrain XTest],dim);

%%务必进行归一化,否则精确度低2个点

X =  normalize(X,2,"norm");


%%
end