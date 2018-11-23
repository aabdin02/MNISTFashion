Use a fixed number of layers: 1 - S^n - 1
Hyper-parameter:
- Learning Rate
    Too high - overshoot the minimum (overfitting)
    Too low - too many iterations to get to the minimum
- Number of neurons in hidden-layer
- Number of epochs of training
- Mini-Batch size

Gradient Descent (GD) requires a cost-function, a gradient (dj/dw - derivative of 
cost-function).
Due to the size of our inputs, experiment with Batch-GD: to get a smooth graph

Goals/Experiments:
1. Determine the best learning rate 
 - Keep the epoches constant of 200, number of neurons in the hidden layer 200 
   and batch-size of 1.
 - Determined that the best learning rate is between 0 - 0.1
2. (hidden_nneuron) Determine the number of a neuron 
 - Using the learning rate of 0.06, determine the best number of neurons in hidden layer
    epoches of 200, mini-batch size of 1
3.(randomTest) Try number of neurons in hidden layer = 65, learning-rate = 0.0001, epoches = 10
input = 100
3. Determine the number of epoches

4. Determine the size of a batch-size





