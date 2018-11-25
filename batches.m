function correct = batches(P,T,l_rate, nneurons,epoches)
   [IRow,~] = size(P);
    
    n_test = 200;
    %Between [0 - 100]
    b_sizes = randperm(300,n_test) ;
    accurracy = zeros(n_test,1);
    index = 1;

    %backprogation(P,T,l_rate,nneuron, epoches)
    for b_size = b_sizes
         correct = backprogation(P',T,l_rate,nneurons, epoches,b_size);
        accurracy(index) = correct / IRow;
        index = index + 1;
    end
    
    scatter(b_sizes', accurracy');
    xlabel('Batch Size');
    ylabel('Percent Accuracy');
    title('Number of Batch Size VS Percent Accuracy');