---
published: true
---
## Intro
In this post, I'm going to explain how we can build a Logistic Regression machine learning algorithm using Python from scratch. I'm also going to explain the concept of this algorithm and discuss the math involved.

## Definition
First of all, what is Logistic Regression? In simple term, Logistic Regression is an extention to the [Linear Regression](https://allen-q.github.io/Implement-Linear-Regression-in-Python-From-Scratch) algorithm we have discussed earlier. A Linear Regression model's output is continous in either directions. It's useful to predict things like house price, stock price, sales etc. However, it's not suitble to predict categorical output like Positive/Negative, True/False or Win/Loss etc. Logistic Regression extends Linear Regression by applying the logistic function to the output such that the output is squashed into the range of 0 and 1. The output can be intepreted as the posibility of something happening, or it can be easily transformed to a categorical ouput by setting a threshold. For example, if the output value is greater than 0.5, we transform it to Positive, otherwise Negative. 

The logistic function can take any real input $$t$$, ($$t \in R$$), whereas the output always takes values between 0 and 1. The logistic function is defined as follows:

\\[\sigma(t)=\frac{1}{1+e^{-t}}\\]

For Logistic Regression, $$t$$ is the output of Linear Regression $$W^T*X$$, therefore, the formular can be transformed as below:

$$\hat{P}(y=+1|W,X)=\frac{1}{1+e^{-(W^T*X)}}$$

This function gives the probability of Y being 1 givin W and X.

## Quality Metric

Like Linear Regression where we used a loss function to measure the quality of the estimated parameters W, we need to find a quality metric for Logistic Regression. 

We used MSE as the cost function(shown below) for Linear Regression. 

\\[ L=\frac{1}{n} * \sum_{i=1}^n(\hat{Y}-Y)^2\\]

Howerver, MSE is not suitable for Logistic Regression because the function turns out to be non-convex. A non-convex function has multiple local minima which means we can't use gradient descent to minimize the cost function.

Instead we can go back to the definition of the Logistic Regression Function. Remember the output of the function is the probability of Y being 1. If we have a perfect classifier, it will output 1 for all data 
points with label 1 and output 0 for all datapoints with label 0 like below:
        
$$\hat{P} = \left\{
         \begin{array}{l l}
            1 & \quad \text{if $y$ = 1}\\
            0 & \quad \text{if $y$ = 0}\\
          \end{array} 
          \right.$$

If we assume all the data points are independent, the probability of the entire data set is the product of the probabilities of the individual data points:

$$\ell(W)=\prod_{i=1}^NP(y_{i}|W,X_{i})$$

For a perfect classifier which classifies everything right, the output of this function will be 1. On the contrary, for the worst classifier which doesn't classify anything right, the output of this function will be 0. In another word, the higher the output of this function, the better our classifier is. This function is called likelihood function and we can use it as our quality metric to measure the performance of our classifier. Unfortunately there is no closed-form solution to solve this function so we have to use other optimization methods such as gradient descent to find the maximum. 

One little trick with maximize the likelihood function is to do a log transformation first. 

$$\ell\ell(W)=\lg{\prod_{i=1}^NP(y_{i}|W,X_{i})}$$


