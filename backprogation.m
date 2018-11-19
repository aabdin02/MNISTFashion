function correct = backprogation(P,T,l_rate,nneuron, epoches)
    % Initializing variables
    correct = 0;
    [IRow,~] = size(P);
    [~,TCol] = size(T);

    %Random Weight Generation:
    W1 = zeros(nneuron, IRow);
    b1 = zeros(nneuron,1);

    W2 = zeros(TCol,nneuron);
    b2 = zeros(TCol,1);

    %For Layer 1
    for row = 1: nneuron
        for col = 1: IRow
            W1(row,col) = rand(1);
        end
        b1(row,1) = rand(1);
    end
    %For Layer 2
    for row = 1: TCol
        for col = 1: nneuron
            W2(row,col) = rand(1);
        end
        b2(row,1) = rand(1);
    end
    
    %Forward Propagation
    for epoch = 0: epoches
        t = 0;
        for p = P
            t = t + 1;
            %1. Forward Propagation
            a0 = P(:,t);
            a1 = satlin(W1 * a0 + b1);
            a2 = round(poslin(W2 * a1 + b2));

            %2. Error Calculation
            error = T(t,:)- a2;

            %3. Backward Propagation (Sentivity Calculations)
            %Calculating Sentivity for layer #2
            F2deri = dposlin(a2);
            s2 = -2 * diag(F2deri) *error;
            % Sentivity for layer #1
            F1deri = dsatlin(a1);
            s1 = diag(F1deri) * W2' * s2;

            %4. Approximate Steepest Descent
            % Weight Update
            W2 = W2 - l_rate * s2 * a1';
            W1 = W1 - l_rate * s1 * a0';
            % Bias Update
            b2 = b2 - l_rate * s2;
            b1 = b1 - l_rate * s1;
           
        end
    end
    
    %For Determining the number of correct output
    t = 0;
    for p = P
        a0 = p;
        a1 = satlin(W1 * a0 + b1);
        a2 = round(poslin(W2 * a1 + b2));
        t = t + 1;
        error = T(t,:)- a2
        if error == 0
            correct = correct + 1;
        end
        
    end