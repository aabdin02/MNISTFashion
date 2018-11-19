%Pattern Classification:
P = xlsread('train13.xlsx'); %(R,C) R = number of input C = size of inputs
P(:,1) = [] ;%Remove the Indexs
T = P(:,1)  ;
P(:,1) = [] ;%Remove the targets

[IRow,~] = size(P);
epoches = 200; %zeros(20);
nneuron = 200;

%Between [0 - 1]
n_test = 100;
l_rates = rand(1,n_test);
accurracy = zeros(n_test);
index = 1;

%backprogation(P,T,l_rate,nneuron, epoches)
for l_rate = l_rates
    correct = backprogation(P',T,l_rate,nneuron, epoches);
    accurracy(index) = correct / IRow;
    index = index + 1;
end
plot(l_rates,accurracy, 'b-o', 'LineWidth',1);
xlabel('Learning Rate');
ylabel('Percent Accuracy');
legend('0 pixels');
title('Learning Rates VS Percent Accuracy');

%Between [0 - 100]
l_rates = randperm(100,n_test);
accurracy = zeros(n_test);
index = 1;
correct = 0;
%backprogation(P,T,l_rate,nneuron, epoches)
for l_rate = l_rates
    correct = backprogation(P',T,l_rate,nneuron, epoches);
    accurracy(index) = correct / IRow;
    index = index + 1;
end
figure(2);
plot(l_rates,accurracy, 'b-o', 'LineWidth',1);
xlabel('Learning Rate');
ylabel('Percent Accuracy');
legend('0 pixels');
title('Learning Rates VS Percent Accuracy');
