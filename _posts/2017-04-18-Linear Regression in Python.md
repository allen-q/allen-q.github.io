---
published: true
---

In this post, I'm going to implement the Linear Regression algorithm from scrach using pure Python code.


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
    dW = np.mean((y_pred - y_data)*X_data)
    db = np.mean(y_pred - y_data)    
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
