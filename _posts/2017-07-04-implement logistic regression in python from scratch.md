---
published: true
---
## Intro
In this post, I'm going to explain how we can build a Logistic Regression machine learning algorithm using Python from scratch. I'm also going to explain the concept of this algorithm and discuss the math involved.

## Definition
First of all, what is Logistic Regression? In simple term, Logistic Regression is an extention to the [Linear Regression](https://allen-q.github.io/Implement-Linear-Regression-in-Python-From-Scratch) algorithm we have discussed earlier. A Linear Regression model's output is continous in either directions. It's useful to predict things like house price, stock price, sales etc. However, it's not suitble to predict categorical output like Positive/Negative, True/False or Win/Loss etc. Logistic Regression extends Linear Regression by applying the logistic function to the output such that the output is squashed into the range of 0 and 1. The output can be intepreted as the posibility of something happening, or it can be easily transformed to a categorical ouput by setting a threshold. For example, if the output value is greater than 0.5, we transform it to Positive, otherwise Negative. 

The logistic function can take any real input $$t$$, ($$t \in R$$), whereas the output always takes values between 0 and 1. The logistic function is defined as follows:

\\[\sigma(t)=\frac{1}{1+e^{-t}}\\]

For Logistic Regression, $$t$$ is the output of Linear Regression $$W*X+b$$, therefore, the formular can be transformed as below:

\\[F(x)=\frac{1}{1+e^{-(W*X+b)}}\\]

## Cost Function
We used MSE as the cost function(shown below) for Linear Regression. 

\\[ L=\frac{1}{n} * \sum_{i=1}^n(\hat{Y}-Y)^2\\]

Howerver, MSE is not suitable to be used as a cost function for Logistic Regression because the function turns out to be non-convex. A non-convex function has multiple local minima which means we can't use gradient descent to minimize the cost function.

Instead we define the cost function for Logistic Regression as below:

\\[F(x) = \left\{ 
         \begin{array}{l l}
            -log(F(x)) & \quad \text{if $y$ = 1}\\
            -log(1-F(x)) & \quad \text{if $y$ = 0}\\
          \end{array} 
          \right.\\]
          
$$F(x) = \left\{
         \begin{array}{l l}
            -log(F(x)) & \quad \text{if $y$ = 1}\\
            -log(1-F(x)) & \quad \text{if $y$ = 0}\\
          \end{array} 
          \right.$$
          
\\[F(x) = \left\{
         \begin{array}{l l}
            -log(F(x)) & \quad \text{if $y$ = 1}\\
            -log(1-F(x)) & \quad \text{if $y$ = 0}\\
          \end{array} 
          \right.\\]



