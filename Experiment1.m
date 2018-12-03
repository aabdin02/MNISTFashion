%Pattern Classification:
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

Xtrain = inputs(:, training(c));
Ytrain = targetsd (:, training(c));
Xtest = inputs(:, test(c));
Ytest = targets(test(c));
Ytestd = targetsd(:, test(c));


Ypred = myNNfun(Xtest);
Ypred(:,1:5);
[~,Ypred] = max(Ypred);
sum(Ytest == Ypred) / length(Ytest);

sweep = [250];
scores = zeros(length(sweep), 1);
models = cell(length(sweep), 1);

x = Xtrain;
t = Ytrain;
trainFcn = 'logsig';

for i = 1: length(sweep)
    hiddenLayerSize = sweep(i);
    net = patternnet(hiddenLayerSize);
    net.divideParam.trainRatio = 0.7;
    net.divideParam.valRatio = 0.15;
    net.divideParam.testRatio = 0.15;
    net = train(net, x, t);
    models{i} = net;
    p = net(Xtest);
    [~,p] = max(p);
    scores(i) = sum(Ytest == p) / length(Ytest);
end


n = size(sub,1);
sub = sub';
[~,highest] = max(scores);