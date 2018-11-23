function epoches = epoches(P,T,l_rate, nneurons)
    [IRow,~] = size(P);
    
    n_test = 200;
    %Between [0 - 100]
    epoches = randperm(300,n_test) ;
    accurracy = zeros(n_test,1);
    index = 1;

    %backprogation(P,T,l_rate,nneuron, epoches)
    for epoch = epoches
        correct = backprogation(P',T,l_rate,nneurons, epoch);
        accurracy(index) = correct / IRow;
        index = index + 1;
    end
    
    scatter(epoches', accurracy');
    xlabel('Number of epoches');
    ylabel('Percent Accuracy');
    title('Number of Epoches VS Percent Accuracy');