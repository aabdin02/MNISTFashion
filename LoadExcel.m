%Pattern Classification:
P = xlsread('train13.xlsx'); %(R,C) R = number of input C = size of inputs
P(:,1) = [] ;%Remove the Indexs
T = P(:,1)  ;
P(:,1) = [] ;%Remove the targets

%Experiment #1: Use 0.01 as the learning rate
%learning_rate(P,T);
l_rate = 0.06;
%Experiment #2: Determine the number of neurons in the hidden layer
nneurons = hidden_nneuron(P,T, l_rate); 
