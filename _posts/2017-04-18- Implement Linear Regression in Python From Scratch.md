---
published: true
---

In this post, I'm going to explain how we can build a simple Linear Regression machine learning algorithm using Python from scratch. 

First let's create some toy data set.
```python
import numpy as np

#generate some test data
X_data = np.arange(100,step=0.1, dtype=np.float32)

#add a bit gaussian noise to Y
y_data = (4000 + 50 * X_data) + np.random.normal(0,3,size=len(X_data))*100

#plot raw data
print('Raw Data')
plt.plot(X_data,y_data)
plt.show()
```

Plot the raw data:

![k-means_python-iter5]({{site.baseurl}}/images/lr/output_0_1.png)

A single variable linear function has an equation of the form $$ Y = W*X + b $$ where X is the independent variable and Y is the dependent variable. The slope of the line is W and b is the intercept.

For the sample data set we have generated, we know W is about 50 and b is about 4000. However, in real world when we get a dataset, these parameters are often unknown and that's the problem we are going to solve. We are going to build a simple machine learning model to 'learn' these parameters from the training data set. 

First step is to initilize the parameters W and b. In this example, I'm going to assign a random number to W and 0 to b. As long as you assign a non-zero value to W the model should work. It might just take longer to converge with a bad initialization.

```python
W = np.random.random()
b = 0
```

We can then estimate Y using the the formular $$ Y = W*X + b $$

```python
y_pred = X_data * W + b
```

The next step is to determine how close y_pred is to the ground truth y_data. To quantify how good our y_pred, we introduce a metric called 'loss function' denoted as L. There are many ways to define a loss function and one of the most commonly used loss function is called the Mean Square Error (MSE). MSE is defined as $$ MSE = \frac{1}{n} * \sum_{i=1}^n(\hat{Y}-Y)^2 $$. Below is the python code to calculate loss.


```python
L = 0.5*np.mean((y_pred - y_data)**2)  
```

We now know how good the W and b parameters fit the data set, we need to find a way to improve them. We will do it by using an algorithm called gradient descent. 

Gradient descent is an iterative approach to finding the minimum of the loss function by changing X along the negative direction of the gradient of the loss function. This process is similar to walking downwards a hill. The gradient of the hill is the steepest direction at a point to the top. As we need to walk downwards, we take opposite direction of the gradient. By repeating this process, it can guarantee we reach the lowest point after some iterations. Gradient is a vector of partial derivatives with repect to each of the independent variables, in this case, W and b. So all we need to do is to calculate the partial derivatives of L with repect to W and b and then update W and b in the negative direction of the derivatives. 

Partial derivative of L w.r.t W: 

$$ \frac{\partial L}{\partial b}=\frac{1}{N}*\sum(\widehat{y}-y)*X $$
\\[\frac{\partial L}{\partial b}=\frac{1}{N}*\sum(\widehat{y}-y)\\]

\\[\frac{\partial L}{\partial W}=\frac{1}{N}*\sum((\widehat{y}-y)*X)\\]

Partial derivative of L w.r.t b: 

\\[\frac{\partial L}{\partial b}=\frac{1}{N}*\sum(\widehat{y}-y)\\]

```python
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 14:57:03 2017

@author: allenq
"""

import numpy as np
from seaborn import plt


#generate some test data
X_data = np.arange(100,step=0.1, dtype=np.float32)

#add a bit gaussian noise to Y
y_data = (4000 + 50 * X_data) + np.random.normal(0,3,size=len(X_data))*100

#plot raw data
print('Raw Data')
plt.clf()
plt.plot(X_data,y_data)
plt.show()

#initialize weights
W = np.random.random()
b = 0
step_size = 0.0005
max_iteration = 50000

for epoch in range(max_iteration):    
    y_pred = X_data * W + b
    loss = 0.5*np.mean((y_pred - y_data)**2)  
    # The partial derivative of loss function w.r.t W
    dW = np.mean((y_pred - y_data)*X_data)
    # The partial derivative of loss function w.r.t b
    db = np.mean(y_pred - y_data)   
    # Take one step in the negative direction of the derivative
    W -= step_size*dW
    b -= step_size*db    
    
    if (epoch%(max_iteration/10) == 0):
        print('Epoch:{}, W:{}, b:{}, loss:{}, dW:{}, db:{}'
              .format(epoch,W,b,loss,dW,db))
        plt.clf()
        plt.ylim(0, max(y_data)*1.2)
        plt.plot(X_data,y_data)
        plt.plot(X_data,X_data * W + b)
        plt.show()
        
print('Training Stats: \nEpochs:{}, W:{}, b:{}, loss:{}, dW:{}, db:{}'.format(epoch,W,b,loss,dW,db))
```

    Raw Data
    


![png]({{site.baseurl}}/images/lr/output_0_1.png)


    Epoch:0, W:182.87746440940936, b:3.2356393868964397, loss:22003630.303832497, dW:-364407.97873254475, db:-6471.278773792879
    


![png]({{site.baseurl}}/images/lr/output_0_3.png)


    Epoch:5000, W:82.19436206027052, b:1863.6071594519628, loss:621539.4658909226, dW:8.044985679682345, db:-535.7701051391163
    


![png]({{site.baseurl}}/images/lr/output_0_5.png)


    Epoch:10000, W:67.25272119607219, b:2859.1438826720114, loss:212077.28846892182, dW:4.297215674001723, db:-286.5374256957569
    


![png]({{site.baseurl}}/images/lr/output_0_7.png)


    Epoch:15000, W:59.26171428220094, b:3391.5709008278927, loss:94960.464222252, dW:2.3121905121668243, db:-153.24401749263194
    


![png]({{site.baseurl}}/images/lr/output_0_9.png)


    Epoch:20000, W:54.988020624082026, b:3676.3203462049905, loss:61462.02017007064, dW:1.2326599906920455, db:-81.9571420043507
    


![png]({{site.baseurl}}/images/lr/output_0_11.png)


    Epoch:25000, W:52.70238928113305, b:3828.6083480619427, loss:51880.58116044198, dW:0.6435117712257197, db:-43.832059973100705
    


![png]({{site.baseurl}}/images/lr/output_0_13.png)


    Epoch:30000, W:51.47998993561014, b:3910.054109952339, loss:49140.04065844567, dW:0.37040056447545067, db:-23.441584631303822
    


![png]({{site.baseurl}}/images/lr/output_0_15.png)


    Epoch:35000, W:50.82624364074737, b:3953.6124468015896, loss:48356.17221420905, dW:0.19513092788661016, db:-12.536938879350698
    


![png]({{site.baseurl}}/images/lr/output_0_17.png)


    Epoch:40000, W:50.47660880698521, b:3976.908057570531, loss:48131.96822914358, dW:0.10178082272975734, db:-6.704953283647574
    


![png]({{site.baseurl}}/images/lr/output_0_19.png)


    Epoch:45000, W:50.289622654662296, b:3989.3668778007655, loss:48067.84134583347, dW:0.03796690074479556, db:-3.586184728960075
    


![png]({{site.baseurl}}/images/lr/output_0_21.png)


    Training Stats: 
    Epochs:49999, W:50.189632905662165, b:3996.0290701946233, loss:48049.49881550643, dW:0.018013466784948834, db:-1.9182123168506997
    


```python

```
