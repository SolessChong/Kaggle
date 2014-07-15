% Load data
%[X, Y, test] = loadData();
Yexp = expandY(Y);

nTrial = 12;
error_array = zeros(nTrial,1);

parfor k = 1 : nTrial
    CVO = cvpartition(Y,'holdout',1/3);
    err = zeros(CVO.NumTestSets,1);
    for i = 1 : CVO.NumTestSets
        trInd = CVO.training(i);
        teInd = CVO.test(i);

        [labels, nn] = predict_SAE(X(trInd,:), Yexp(trInd,:), X(teInd,:));

        err(i,1) = sum(Y(teInd) ~= labels);
    end
    
    error_array(k) = 1 - sum(err)/CVO.TestSize;
    
    disp(error_array(k));
end

disp(error_array);

%labels = predict_SAE(X, Yexp, test);

%genSubmission(labels, 'SAE');