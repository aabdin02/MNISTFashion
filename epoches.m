function best_epoche = epoches(P,T,l_rate,W1,W2,b1,b2,batch_size)
    best_epoche = 0;
    n_test = 20;
    %Between [0 - 100]
    epoche = randperm(200,n_test) ;
    accurracy = zeros(n_test,1);
    index = 1;
    [IRow,~] = size(P);
    
    
    %backprogation(P,T,l_rate,nneuron, epoches)
    for epoch = epoche
        correct = backprogation(P',T,W1,W2,b1,b2,l_rate,epoch,batch_size);
        if correct > best_epoche
            best_epoche = correct;
        end
        accurracy(index) = correct / IRow;
        index = index + 1;
    end
    
    scatter(epoche', accurracy');
    xlabel('Number of epoches');
    ylabel('Percent Accuracy');
    
    title(sprintf('%s %d','Number of Epoches VS Percent Accuracy With ',batch_size));