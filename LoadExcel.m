%Pattern Classification:
P = xlsread('train13.xlsx'); %(R,C) R = number of input C = size of inputs
P(:,1) = [] ;%Remove the Indexs
T = P(:,1)  ;
P(:,1) = [] ;%Remove the targets

learning_rate(P,T)