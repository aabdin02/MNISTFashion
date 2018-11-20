function l_rate = learning_rate(P,T)    
    [IRow,~] = size(P);
    epoches = 200; %zeros(20);
    nneuron = 200;

    %Between [0 - 1]
    n_test = 100;
    l_rates = rand(1,n_test);
    accurracy = zeros(n_test,1);
    index = 1;

    %backprogation(P,T,l_rate,nneuron, epoches)
    for l_rate = l_rates
        correct = backprogation(P',T,l_rate,nneuron, epoches);
        accurracy(index) = correct / IRow;
        index = index + 1;
    end

    scatter(l_rates,accurracy');
    xlabel('Learning Rate');
    ylabel('Percent Accuracy');
    title('Learning Rates VS Percent Accuracy');

    %Between [0 - 100]
    l_rates = randperm(100,n_test) ;
    accurracy = zeros(n_test,1);
    index = 1;

    %backprogation(P,T,l_rate,nneuron, epoches)
    for l_rate = l_rates
        correct = backprogation(P',T,l_rate,nneuron, epoches);
        accurracy(index) = correct / IRow;
        index = index + 1;
    end

    figure(2);
    
    scatter(l_rates, accurracy');
    xlabel('Learning Rate');
    ylabel('Percent Accuracy');
    title('Learning Rates VS Percent Accuracy');
