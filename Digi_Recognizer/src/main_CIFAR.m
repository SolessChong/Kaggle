% Load data
%[X, Y, test] = loadData();

load('D:\Workspace\Kaggle\Digi_Recognizer\data\CIFAR\data_batch_1.mat')
data_all = data;
labels_all = labels;
load('D:\Workspace\Kaggle\Digi_Recognizer\data\CIFAR\data_batch_2.mat')
data_all = [data_all; data]; labels_all = [labels_all; labels];
load('D:\Workspace\Kaggle\Digi_Recognizer\data\CIFAR\data_batch_3.mat')
data_all = [data_all; data]; labels_all = [labels_all; labels];
load('D:\Workspace\Kaggle\Digi_Recognizer\data\CIFAR\data_batch_4.mat')
data_all = [data_all; data]; labels_all = [labels_all; labels];
load('D:\Workspace\Kaggle\Digi_Recognizer\data\CIFAR\data_batch_5.mat')
data_all = [data_all; data]; labels_all = [labels_all; labels];

data = data_all;
labels = labels_all;

Y = labels;
X = data;
Yexp = expandY(labels);

nTrial = 1;
error_array = zeros(nTrial,1);

for k = 1 : nTrial
    CVO = cvpartition(Y,'holdout',1/5);
    err = zeros(CVO.NumTestSets,1);
    for i = 1 : CVO.NumTestSets
        trInd = CVO.training(i);
        teInd = CVO.test(i);

        %[labels, nn{k}, sae{k}] = predict_SAE(X(trInd,:), Yexp(trInd,:), X(teInd,:));
        [labels, cnn{k}] = predict_CNN_CIFAR(X(trInd,:), Yexp(trInd,:), X(teInd,:));

        err(i,1) = sum(Y(teInd) ~= labels);
    end
    
    error_array(k) = 1 - sum(err)/CVO.TestSize;
    
end

disp(error_array);
disp(mean(error_array));

%labels = predict_SAE(X, Yexp, test);

%genSubmission(labels, 'SAE');