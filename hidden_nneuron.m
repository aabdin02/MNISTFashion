function nneuron = hidden_nneuron(P,T,l_rate)
    [IRow,~] = size(P);
    epoches = 10; %zeros(20);
    n_test = 200;
    %Between [0 - 100]
    nneurons = randperm(300,n_test) ;
    accurracy = zeros(n_test,1);
    index = 1;

    %backprogation(P,T,l_rate,nneuron, epoches)
    for nneuron = nneurons
        correct = backprogation(P',T,l_rate,nneuron, epoches);
        accurracy(index) = correct / IRow;
        index = index + 1;
    end
    
    scatter(nneurons', accurracy');
    xlabel('Number of Neurons');
    ylabel('Percent Accuracy');
    title('Number of Neurons VS Percent Accuracy');