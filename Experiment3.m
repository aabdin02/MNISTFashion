sub = csvread('test.csv',1,1);
n = size(sub,1) + 60000;
sub = sub';
sub = reshape(sub, 28,28,1,[]);


Ypred  = predict(net,sub)';
[~,label] = max(Ypred);

label = label';
Id = 60001:n; Id = Id';
writetable(table(Id, label), 'submission.csv');
