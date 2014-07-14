% Load data
%[X, Y, test] = loadData();
Yexp = expandY(Y);

nTrial = 5;
error_array = zeros(nTrial,1);

for k = 1 : nTrial
    CVO = cvpartition(Y,'holdout',1/3);
    err = zeros(CVO.NumTestSets,1);
    for i = 1 : CVO.NumTestSets
        trInd = CVO.training(i);
        teInd = CVO.test(i);

        labels = predict_SAE(X(trInd,:), Yexp(trInd,:), X(teInd,:));
        Yshr = shrinkY(Yexp);
        err(i,1) = sum(Yshr(teInd) ~= labels);
    end
    
    error_array(k) = sum(err)/CVO.TestSize;
    
    disp(error_array(k));
end