function [labels, cnn] = predict_CNN(X, Y, test)
%
% Predict via CNN
%
% Train and predict
%
global config

train_x = double(reshape(X', 28, 28, size(X,1))) / 255;
test_x = double(reshape(test', 28, 28, size(test,1))) / 255;
train_y = double(Y');

%% Train a 6c-2s-12c-2s Convolutional neural network 
%will run 1 epoch in about 200 second and get around 11% error. 
%With 100 epochs you'll get around 1.2% error
cnn.layers = {
    struct('type', 'i')
    struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5); % conv. layer
    struct('type', 's', 'scale', 2) % sub sampling layer
    struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5)
    struct('type', 's', 'scale', 2)
    };
cnn = cnnsetup(cnn, train_x, train_y);

opts.alpha = 1;
opts.batchsize = 50;
opts.numepochs = 200;
cnn = cnntrain(cnn, train_x, train_y, opts);

labels = cnnpredict(cnn, test_x)' - 1;

% plot MSE
figure; plot(cnn.rL);

end


function [labels, net] = cnnpredict(cnn, X)

net = cnnff(cnn, X);
[~, labels] = max(net.o);

end
