tr = csvread('train.csv',1,1);
sub = csvread('test.csv',1,1);

n = size(tr,1);
targets = tr(:,1);
targets(targets == 0) = 10;
targetsd = dummyvar(targets);
inputs = tr(:,2:end);

inputs = inputs';
targets = targets';
targetsd = targetsd';

rng(1);
c = cvpartition(n, 'Holdout', n/3);

XTrain = inputs(:, training(c));
YTrain = targetsd (:, training(c));

XValidation = inputs(:, test(c));
YValidation = targets(test(c));
Ytestd = targetsd(:, test(c));

layers = [
    imageInputLayer([28 28 1])
    
    convolution2dLayer(3,8, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3, 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(10)
    regressionLayer];
    
options = trainingOptions('sgdm', 'InitialLearnRate', 0.001, ...
    'MaxEpochs', 4, 'Shuffle', 'every-epoch', ...
    'ValidationData', {XValidation,YValidation}, 'ValidationFrequency', 30, ...
    'Verbose', false, 'Plots', 'training-progress');

YTrain = YTrain';
net = trainNetwork(XTrain,YTrain , layers, options);

