function labels = predict_SAE(X, Y, test)
%
% Predict via Stacked Auto-Encoder
%
% Train and predict
%
global config

train_x = double(X)/255;
train_y = double(Y)/255;
test_x = double(test)/255;

%  Setup and train a stacked denoising autoencoder (SDAE)
sae = saesetup([config.w * config.h, 100]);
sae.ae{1}.activation_function       = 'sigm';
sae.ae{1}.learningRate              = 0.5;
sae.ae{1}.inputZeroMaskedFraction   = 0.5;
opts.numepochs =   10;
opts.batchsize = 100;
sae = saetrain(sae, train_x, opts);
visualize(sae.ae{1}.W{1}(:,2:end)')

% Use the SDAE to initialize a FFNN
nn = nnsetup([config.w * config.h, 100, 10]);
nn.activation_function              = 'sigm';
nn.learningRate                     = 0.5;
nn.W{1} = sae.ae{1}.W{1};

% Train the FFNN
opts.numepochs =   10;
opts.batchsize = 100;
nn = nntrain(nn, train_x, train_y, opts);

labels = nnpredict(nn, test);