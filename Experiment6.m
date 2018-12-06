%Pattern Classification:
tr = csvread('train.csv',1,1);
%sub = csvread('test.csv',1,1);

n = size(tr,1);
targets = tr(:,1);
targets(targets == 0) = 10;
targetsd = dummyvar(targets);
inputs = tr(:,2:end);

inputs = inputs';
targets = categorical(targets');
targetsd = targetsd';

rng(1);
c = cvpartition(n, 'Holdout', n/3);

%XTrain = inputs(:, training(c));
YTrain = targets(training(c));

YTrain = categorical(YTrain);

XValidation = inputs(:, test(c));
XValidation = reshape(XValidation, 28,28,1, []);

Ytestd = targets( test(c));
%Ytestd = targetsd(:, test(c))';
YValidation = categorical(Ytestd');

numTrainImages = numel(YTrain);

XTrain = reshape(XTrain, 28,28,1, []);
idx = randperm(numTrainImages,20);


layers = [
   imageInputLayer([28 28 1])

    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    averagePooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    averagePooling2dLayer(2,'Stride',2)
  
    %convolution2dLayer(3,32,'Padding','same')
    %batchNormalizationLayer
    %reluLayer
    
    %averagePooling2dLayer(2,'Stride',2)
  
    dropoutLayer(0.2)
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    dropoutLayer(0.2)
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer
    %regressionLayer
    ];


miniBatchSize  = 27;
validationFrequency = floor(numel(YTrain)/miniBatchSize);

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.0001, ...
    'MaxEpochs',42, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{XValidation,YValidation}, ...
    'ValidationFrequency',validationFrequency, ...
    'Verbose',false, ...
    'Plots','training-progress');


net = trainNetwork(XTrain,YTrain,layers,options);

%Use Experiment3 for completing the submission.csv

