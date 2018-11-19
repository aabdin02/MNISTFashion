%Pattern Classification:
P = xlsread('train13.xlsx'); %(R,C) R = number of input C = size of inputs
P(:,1) = [] ;%Remove the Indexs
T = P(:,1)  ;
P(:,1) = [] ;%Remove the targets
l_rate = 0.06;

correct = backprogation(P',T,0.06);
trials = 5;

while correct < 50

    correct = 0;
    for trial = 0: trials
        correct = correct + backprogation(P',T, l_rate);
    end
    
    l_rate
    correct = correct / trials
end
