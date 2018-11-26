function labels = predict(PTest,W1,W2,b1,b2)
%For Determining the number of correct output
    t = 0;
    labels = zeros(10,1);
    for p = PTest
        a0 = p;
        a1 = satlin(W1 * a0 + b1);
        a2 = round(poslin(W2 * a1 + b2));
        t = t + 1;
        labels(t) = a2;
    end