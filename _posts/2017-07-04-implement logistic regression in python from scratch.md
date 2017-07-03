---
published: true
---
In this post, I'm going to explain how we can build a Logistic Regression machine learning algorithm using Python from scratch. I'm also going to explain the concept of this algorithm and discuss the math involved.

First of all, what is Logistic Regression? In simple term, Logistic Regression is an extention to the [Linear Regression](https://allen-q.github.io/Implement-Linear-Regression-in-Python-From-Scratch) algorithm we have discussed earlier. A Linear Regression model's output is continous in either directions. It's useful to predict things like house price, stock price, sales etc. However, it's not suitble to predict categorical output like Positive/Negative, True/False or Win/Loss etc. Logistic Regression extends Linear Regression by applying the logistic function to the output such that the output is squashed into the range of 0 and 1. The output can be intepreted as the posibility of something happening, or it can be easily transformed to a categorical ouput by setting a threshold. For example, if the output value is greater than 0.5, we transform it to Positive, otherwise Negative. 

The logistic function can take any real input $$t$$, ($$t\niR$$), whereas the output always takes values between zero and one and hence is interpretable as a probability. The logistic function is defined as follows: