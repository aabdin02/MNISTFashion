%Pattern Classification:
tr = csvread('train.csv',1,1);
%sub = csvread('test.csv',1,1);

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
YTrain = targetsd(:, training(c));

XValidation = inputs(:, test(c));
XValidation = reshape(XValidation, 28,28,1, []);

Ytestd = targetsd(:, test(c));
YValidation = Ytestd';

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
  
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    dropoutLayer(0.2)
    fullyConnectedLayer(10)
    %softmaxLayer
    %classificationLayer
    regressionLayer
    ];


miniBatchSize  = 27;
validationFrequency = floor(numel(YTrain)/miniBatchSize);

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{XValidation,YValidation}, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

%YTrain = num2cell(YTrain,1);
YTrain = YTrain';

net = trainNetwork(XTrain,YTrain,layers,options);


YPred = predict(net,XValidation);
accuracy = sum(YPred == YValidation)/numel(YValidation);