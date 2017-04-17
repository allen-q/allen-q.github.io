---
published: true
---
In this post, I'm going to implement a vanilla version of the K-Means algorithm in pure Python and in Tensorflow.

```python
import tensorflow as tf
import pandas as pd
import numpy as np
from seaborn import plt

def plot_labels():
    print("centroids:{}".format(centroids))
    plt.clf()
    for label in range(K):
        plt.scatter(X_data[assignment==label][:,0],
                    X_data[assignment==label][:,1],
                    color=colors[label])
    plt.show()      

def refine_centroids(data,assignment):
    new_centroids = np.zeros((K,2))
    for label in range(K):
        label_data = data[assignment==label]
        dist = (label_data[:,None] - label_data)
        dist_to_others = np.sqrt(np.sum(np.square(dist),axis=-1))
        new_centroids[label] = label_data[np.argmin(np.sum(dist_to_others,axis=-1))] 
    return new_centroids
    
    
def assign_lables(data,centroids):  
    dist = data[:,np.newaxis]-centroids
    assignment = np.argmin(np.sqrt(np.sum(np.square(dist),axis=-1)),axis=-1)
    return assignment
    

X_data = np.concatenate([np.random.normal(i**2,i/2,(512,2)) for i in range(1,7)])
plt.scatter(X_data[:,0],X_data[:,1])
K = 6
colors = np.random.random((K,3))
#inital centroids and assignment
centroids = X_data[np.random.choice(len(X_data),size=K,replace=False)]
assignment=assign_lables(X_data,centroids)
plot_labels()

for i in range(15):    
    centroids = refine_centroids(X_data,assignment)
    assignment = assign_lables(X_data,centroids)
    plot_labels()
```


![k-means-python-initial]({{site.baseurl}}/images/qt_img127951370715140.png)
![k-means_python-iter1]({{site.baseurl}}/images/qt_img127998615355396.png)
![k-means_python-iter3]({{site.baseurl}}/images/qt_img128118874439684.png)
![k-means_python-iter5]({{site.baseurl}}/images/qt_img128239133523972.png)
![k-means_python-iter7]({{site.baseurl}}/images/qt_img128359392608260.png)
![k-means_python-iter9]({{site.baseurl}}/images/qt_img128479651692548.png)
![k-means_python-iter15]({{site.baseurl}}/images/qt_img129141076656132.png)![qt_img129141076656132.png]({{site.baseurl}}/images/qt_img129141076656132.png)



