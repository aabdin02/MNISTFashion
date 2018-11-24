%Pattern Classification:
size = 2000;
P = xlsread('train13.xlsx'); %(R,C) R = number of input C = size of inputs
P = P(1:size,:);
P(:,1) = [] ;%Remove the Indexs
T = P(:,1)  ;
P(:,1) = [] ;%Remove the targets

%Experiment #1: Use 0.01 as the learning rate
%learning_rate(P,T);
l_rate = 0.0001;
%Experiment #2: Determine the number of neurons in the hidden layer
%nneurons = hidden_nneuron(P,T, l_rate); 

epoches(P,T,l_rate,64);

 %For Determining the number of correct output
