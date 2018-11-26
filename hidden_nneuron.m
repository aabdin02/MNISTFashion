function best_nneuron = hidden_nneuron(P,T,l_rate,W1,W2,b1,b2,batch_size)
    [IRow,~] = size(P);
    n_test = 100;
    %Between [0 - 100]
    nneurons = randperm(100,n_test) ;
    accurracy = zeros(n_test,1);
    index = 1;
    best_nneuron = 0;
    
    %backprogation(P,T,l_rate,nneuron, epoches)
    for nneuron = nneurons
        correct = backprogation(P',T,W1,W2,b1,b2,l_rate,epoch,batch_size);
        if correct > best_nneuron
            best_nneuron = correct;
        end
        accurracy(index) = correct / IRow;
        index = index + 1;
    end
    
    scatter(nneurons', accurracy');
    xlabel('Number of Neurons');
    ylabel('Percent Accuracy');
    title('Number of Neurons VS Percent Accuracy');