# Machine Learning Lab

## Worksheet 04 - Gradient descent

### Problem 01

Consider `f(x)=x2-4X`, then we could easily solve to find that  minimizes `f(x)`.

First generate some training data in the form of `(X, Y)` pairs.
You can use `np.random.rand` and `p.random.randn` for this.

Set your `random seed` in numpy to `3` before generating the data so that the results will be comparable.

Plot the data using matplotlib.

Implement the gradient descent algorithm and find `x` value. Plot the cost over epochs for three different learning rates, `0.1`, `0.01`, `0.001`

#### Algorithm 

A cost function, `J`.

This tells you how far you are from the desired result. This is like the difference between the expected output and the actual output in the perceptron algorithm.

Calculate the cost by creating a function which takes three arguments:
- The weights, `W`,
- the inputs `X`,
- the expected outputs `y`

First create the predicted output `ŷ` by taking the dot product of the input and the weights.

Then calculate the cost using the following formula:

`cost = (1/2m) * Σ (ŷ - y)²`

Where,
- `m` is the number of training examples

The gradient descent algorithm is then:

for each epoch calculate the prediction, `ŷ`
update the weights using the gradient of the cost function

To update the weights, you subtract the multiplication of the learning rate by the partial derivative `(dJ/dW)` of the weights.

`W = W - (α * (dJ/dW))`

The learning rate you should be familiar with from the perceptron exercise.

### Problem 02
Update your implementation to use stochastic gradient descent

### Problem 03
Update your implementation to use minibatch gradient descent. Use a batch size of `5` to start out with.

---
