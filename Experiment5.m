P = csvread('train.csv',1,1);
T = P(:,1);
P(:,1) = [];

T(T == 0) = 10;
T = dummyvar(T);

nneurons = 64;
l_rate = 0.001;
outputs = 10;
inputs = 784;

W1 = zeros(nneurons,inputs);
W2 = zeros(outputs,1);
b1 = zeros(nneurons,1);
b2 = zeros(outputs,1);
g_ratio = 22/7;

for nneuron = 1: nneurons
    for input = 1: inputs
        W1(nneuron,input) = rand(1) / g_ratio;
    end
    b1(nneuron,1) = rand(1);
end
   
for output = 1: outputs
    for nneuron = 1: nneurons
        W2(output,nneuron) = rand(1)/ g_ratio;
    end
    b2(output,1) = rand(1);
end

iterations = 10;
iteration = 1;
while (iteration < iterations)
    t_index = 1;
    for a0 = P'
        t = T(t_index,:);
        t_index = t_index + 1;

        n1 = W1 * a0 + b1;
        a1 = logsig(n1);

        n2 = W2 * a1 + b2;
        a2 = softmax(n2);


        F1deri = dlogsig(n1,a1);
        F2deri = dsoftmax(n2);
        S2 = -2 * F2deri * (t' - a2);
        S1 = diag(F1deri)* W2' * S2;

        W1 = W1 - l_rate * S1 * a0';
        W2 = W2 - l_rate * S2 * a1';

        b1 = b1 - l_rate * S1;
        b2 = b2 - l_rate * S2;

    end
    iteration = iteration + 1;
end

t_index = 1;
correct = 0;
for a0 = P'
    a1 = logsig(W1 * a0 + b1);
    a2 = logsig(W2 * a1 + b2);
    t = T(t_index,:);
    t_index = t_index + 1;
    
    error = t' - a2;
    if error == 0
        correct = correct + 1;
    end
end
