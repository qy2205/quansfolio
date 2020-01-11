---
title: 'Regression: TensorFlow vs sklearn vs StatsModels'
categories:
  - data science
tags:
  - TensorFlow
  - Regression
date: 2020-01-07 21:06:12
---

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import tensorflow as tf
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

import time

import matplotlib.pyplot as plt
```

* **Testing example**: Real Estate house price data


```python
data = pd.read_csv("regtestdata.csv")
data = data.drop(columns = ["No", 'X1 transaction date'])
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>X2 house age</th>
      <th>X3 distance to the nearest MRT station</th>
      <th>X4 number of convenience stores</th>
      <th>X5 latitude</th>
      <th>X6 longitude</th>
      <th>Y house price of unit area</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>32.0</td>
      <td>84.87882</td>
      <td>10</td>
      <td>24.98298</td>
      <td>121.54024</td>
      <td>37.9</td>
    </tr>
    <tr>
      <td>1</td>
      <td>19.5</td>
      <td>306.59470</td>
      <td>9</td>
      <td>24.98034</td>
      <td>121.53951</td>
      <td>42.2</td>
    </tr>
    <tr>
      <td>2</td>
      <td>13.3</td>
      <td>561.98450</td>
      <td>5</td>
      <td>24.98746</td>
      <td>121.54391</td>
      <td>47.3</td>
    </tr>
    <tr>
      <td>3</td>
      <td>13.3</td>
      <td>561.98450</td>
      <td>5</td>
      <td>24.98746</td>
      <td>121.54391</td>
      <td>54.8</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5.0</td>
      <td>390.56840</td>
      <td>5</td>
      <td>24.97937</td>
      <td>121.54245</td>
      <td>43.1</td>
    </tr>
  </tbody>
</table>
</div>




```python
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
```

#### Sklearn


```python
%time reg = LinearRegression().fit(X, y)
print("coefficient of determination {0}".format(reg.score(X, y)))
print("beta {0} \nintercept {1}".format(reg.coef_, reg.intercept_))
```

    Wall time: 1.99 ms
    coefficient of determination 0.5711617064827398
    beta [-2.68916833e-01 -4.25908898e-03  1.16302048e+00  2.37767191e+02
     -7.80545273e+00] 
    intercept -4945.595113744408
    


```python
ysklearn = reg.predict(X)
r2_score(y.values, ysklearn)
```




    0.5711617064827398



#### Statsmodels


```python
Xsm = sm.add_constant(X, prepend=False)
%time mod = sm.OLS(y, Xsm)
%time res = mod.fit()
```

    Wall time: 997 µs
    Wall time: 996 µs
    

    D:\Anaconda\lib\site-packages\numpy\core\fromnumeric.py:2389: FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
      return ptp(axis=axis, out=out, **kwargs)
    


```python
res.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>    <td>Y house price of unit area</td> <th>  R-squared:         </th> <td>   0.571</td>
</tr>
<tr>
  <th>Model:</th>                        <td>OLS</td>            <th>  Adj. R-squared:    </th> <td>   0.566</td>
</tr>
<tr>
  <th>Method:</th>                  <td>Least Squares</td>       <th>  F-statistic:       </th> <td>   108.7</td>
</tr>
<tr>
  <th>Date:</th>                  <td>Wed, 08 Jan 2020</td>      <th>  Prob (F-statistic):</th> <td>9.34e-73</td>
</tr>
<tr>
  <th>Time:</th>                      <td>14:32:44</td>          <th>  Log-Likelihood:    </th> <td> -1492.4</td>
</tr>
<tr>
  <th>No. Observations:</th>           <td>   414</td>           <th>  AIC:               </th> <td>   2997.</td>
</tr>
<tr>
  <th>Df Residuals:</th>               <td>   408</td>           <th>  BIC:               </th> <td>   3021.</td>
</tr>
<tr>
  <th>Df Model:</th>                   <td>     5</td>           <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>           <td>nonrobust</td>         <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
                     <td></td>                       <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>X2 house age</th>                           <td>   -0.2689</td> <td>    0.039</td> <td>   -6.896</td> <td> 0.000</td> <td>   -0.346</td> <td>   -0.192</td>
</tr>
<tr>
  <th>X3 distance to the nearest MRT station</th> <td>   -0.0043</td> <td>    0.001</td> <td>   -5.888</td> <td> 0.000</td> <td>   -0.006</td> <td>   -0.003</td>
</tr>
<tr>
  <th>X4 number of convenience stores</th>        <td>    1.1630</td> <td>    0.190</td> <td>    6.114</td> <td> 0.000</td> <td>    0.789</td> <td>    1.537</td>
</tr>
<tr>
  <th>X5 latitude</th>                            <td>  237.7672</td> <td>   44.948</td> <td>    5.290</td> <td> 0.000</td> <td>  149.409</td> <td>  326.126</td>
</tr>
<tr>
  <th>X6 longitude</th>                           <td>   -7.8055</td> <td>   49.149</td> <td>   -0.159</td> <td> 0.874</td> <td> -104.422</td> <td>   88.811</td>
</tr>
<tr>
  <th>const</th>                                  <td>-4945.5951</td> <td> 6211.157</td> <td>   -0.796</td> <td> 0.426</td> <td>-1.72e+04</td> <td> 7264.269</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>240.068</td> <th>  Durbin-Watson:     </th> <td>   2.149</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>3748.747</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 2.129</td>  <th>  Prob(JB):          </th> <td>    0.00</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td>17.114</td>  <th>  Cond. No.          </th> <td>2.35e+07</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 2.35e+07. This might indicate that there are<br/>strong multicollinearity or other numerical problems.



#### TensorFlow (manually)


```python
Xminmax = X.apply(lambda x: (x - x.min())/(x.max() - x.min()))
Xminmax = sm.add_constant(Xminmax, prepend=False)
Xnn = tf.constant(Xminmax.values, dtype = tf.float32)
ynn = tf.constant(y, dtype = tf.float32) 
```


```python
def mean_squared_error(Y, y_pred):
    return tf.reduce_mean(tf.square(y_pred - Y))
def mean_squared_error_deriv(Y, y_pred):
    return tf.reshape(tf.reduce_mean(2*(y_pred - Y)), [1, 1])
def h(X, weights, bias):
    return tf.tensordot(X, weights, axes=1) + bias
```


```python
num_epochs = 20
num_samples = X.shape[0]
batch_size = 100
learning_rate = 0.01

dataset = tf.data.Dataset.from_tensor_slices((Xnn, ynn))
# buffer_size , a fixed size buffer from which the next element will be uniformly chosen from
dataset = dataset.shuffle(buffer_size=400).repeat(num_epochs).batch(batch_size)
iterator = dataset.__iter__()
```


```python
num_features = Xnn.shape[1]
weights = tf.random.normal((num_features, 1))
bias = 0
epochs_plot = []
loss_plot = []

for i in range(num_epochs) :
    epoch_loss = []
    for b in range(int(num_samples/batch_size)):
        x_batch, y_batch = iterator.get_next()
        output = h(x_batch, weights, bias) 
        loss = epoch_loss.append(mean_squared_error(y_batch, output ).numpy())
        dJ_dH = mean_squared_error_deriv(y_batch, output)
        dH_dW = x_batch
        dJ_dW = tf.reduce_mean(dJ_dH*dH_dW)
        dJ_dB = tf.reduce_mean(dJ_dH)
        weights -= learning_rate*dJ_dW
        bias -= learning_rate*dJ_dB
    loss = np.array(epoch_loss).mean()
    epochs_plot.append(i + 1)
    loss_plot.append(loss)
#     if i > 3 and (loss_plot[-2] - loss_plot[-1])/loss_plot[-1] < 0.001:
#         print('Loss is {}'.format(loss))
#         print("early stopping")
#         break
    print('Loss is {}'.format(loss))
```

    Loss is 1358.989013671875
    Loss is 980.7584228515625
    Loss is 687.59228515625
    Loss is 522.5646362304688
    Loss is 414.3692321777344
    Loss is 316.5061340332031
    Loss is 297.97625732421875
    Loss is 246.95008850097656
    Loss is 237.31556701660156
    Loss is 206.25967407226562
    Loss is 218.74339294433594
    Loss is 210.38148498535156
    Loss is 192.06008911132812
    Loss is 205.01051330566406
    Loss is 184.3499755859375
    Loss is 203.98345947265625
    Loss is 178.852783203125
    Loss is 185.07608032226562
    Loss is 191.60342407226562
    Loss is 199.62747192382812
    


```python
plt.plot(epochs_plot, loss_plot) 
plt.show()
```

![img](output_16_0.png)

```python
weights
```




    <tf.Tensor: id=2904, shape=(6, 1), dtype=float32, numpy=
    array([[6.0631566],
           [7.881733 ],
           [7.380377 ],
           [6.316416 ],
           [7.174367 ],
           [8.993184 ]], dtype=float32)>




```python
output = h(Xnn, weights, bias) 
```

* compare results


```python
mean_squared_error(y.values, ysklearn)
```




    79.20185189210986




```python
r2_score(y.values, ysklearn)
```




    0.5711617064827398




```python
mean_squared_error(y.values, np.array(output).ravel())
```




    166.152724132257




```python
r2_score(y.values, np.array(output).ravel())
```




    0.10036635535766536



#### TensorFlow.Keras


```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation = 'linear', input_shape=[6])
])
optimizer = tf.keras.optimizers.SGD(0.01)
model.compile(loss = 'mean_squared_error',
              optimizer = optimizer,
              metrics = ['mean_absolute_error', 'mean_squared_error'])
model.fit(Xnn, ynn, epochs = 80)
```

    Train on 414 samples
    Epoch 1/80
    414/414 [==============================] - 0s 597us/sample - loss: 896.6500 - mean_absolute_error: 26.4509 - mean_squared_error: 896.6500
    Epoch 2/80
    414/414 [==============================] - 0s 43us/sample - loss: 295.1897 - mean_absolute_error: 13.3173 - mean_squared_error: 295.1898
    Epoch 3/80
    414/414 [==============================] - 0s 41us/sample - loss: 172.0404 - mean_absolute_error: 9.8855 - mean_squared_error: 172.0404
    Epoch 4/80
    414/414 [==============================] - 0s 36us/sample - loss: 143.9924 - mean_absolute_error: 9.0765 - mean_squared_error: 143.9924
    Epoch 5/80
    414/414 [==============================] - 0s 41us/sample - loss: 136.0952 - mean_absolute_error: 8.8789 - mean_squared_error: 136.0952
    Epoch 6/80
    414/414 [==============================] - 0s 34us/sample - loss: 132.7646 - mean_absolute_error: 8.7889 - mean_squared_error: 132.7646
    Epoch 7/80
    414/414 [==============================] - 0s 39us/sample - loss: 130.2836 - mean_absolute_error: 8.7316 - mean_squared_error: 130.2836
    Epoch 8/80
    414/414 [==============================] - 0s 36us/sample - loss: 127.7822 - mean_absolute_error: 8.6210 - mean_squared_error: 127.7822
    Epoch 9/80
    414/414 [==============================] - 0s 41us/sample - loss: 125.7170 - mean_absolute_error: 8.5893 - mean_squared_error: 125.7170
    Epoch 10/80
    414/414 [==============================] - 0s 39us/sample - loss: 123.6199 - mean_absolute_error: 8.4686 - mean_squared_error: 123.6199
    Epoch 11/80
    414/414 [==============================] - 0s 39us/sample - loss: 121.7806 - mean_absolute_error: 8.3942 - mean_squared_error: 121.7806
    Epoch 12/80
    414/414 [==============================] - 0s 46us/sample - loss: 119.9222 - mean_absolute_error: 8.3017 - mean_squared_error: 119.9222
    Epoch 13/80
    414/414 [==============================] - 0s 43us/sample - loss: 118.4376 - mean_absolute_error: 8.2547 - mean_squared_error: 118.4376
    Epoch 14/80
    414/414 [==============================] - 0s 41us/sample - loss: 116.5488 - mean_absolute_error: 8.1695 - mean_squared_error: 116.5488
    Epoch 15/80
    414/414 [==============================] - 0s 39us/sample - loss: 114.9699 - mean_absolute_error: 8.1040 - mean_squared_error: 114.9699
    Epoch 16/80
    414/414 [==============================] - 0s 36us/sample - loss: 113.4340 - mean_absolute_error: 8.0370 - mean_squared_error: 113.4340
    Epoch 17/80
    414/414 [==============================] - 0s 41us/sample - loss: 112.2024 - mean_absolute_error: 7.9727 - mean_squared_error: 112.2024
    Epoch 18/80
    414/414 [==============================] - 0s 41us/sample - loss: 110.7733 - mean_absolute_error: 7.8930 - mean_squared_error: 110.7733
    Epoch 19/80
    414/414 [==============================] - 0s 41us/sample - loss: 109.5808 - mean_absolute_error: 7.8631 - mean_squared_error: 109.5808
    Epoch 20/80
    414/414 [==============================] - 0s 36us/sample - loss: 108.2208 - mean_absolute_error: 7.7869 - mean_squared_error: 108.2208
    Epoch 21/80
    414/414 [==============================] - 0s 34us/sample - loss: 107.1407 - mean_absolute_error: 7.7396 - mean_squared_error: 107.1407
    Epoch 22/80
    414/414 [==============================] - 0s 34us/sample - loss: 106.1204 - mean_absolute_error: 7.6984 - mean_squared_error: 106.1204
    Epoch 23/80
    414/414 [==============================] - 0s 39us/sample - loss: 105.0330 - mean_absolute_error: 7.6545 - mean_squared_error: 105.0330
    Epoch 24/80
    414/414 [==============================] - 0s 39us/sample - loss: 104.1281 - mean_absolute_error: 7.6091 - mean_squared_error: 104.1281
    Epoch 25/80
    414/414 [==============================] - 0s 34us/sample - loss: 103.1202 - mean_absolute_error: 7.5681 - mean_squared_error: 103.1202
    Epoch 26/80
    414/414 [==============================] - 0s 34us/sample - loss: 102.1523 - mean_absolute_error: 7.5238 - mean_squared_error: 102.1523
    Epoch 27/80
    414/414 [==============================] - 0s 36us/sample - loss: 101.2129 - mean_absolute_error: 7.4438 - mean_squared_error: 101.2129
    Epoch 28/80
    414/414 [==============================] - 0s 31us/sample - loss: 100.4525 - mean_absolute_error: 7.3894 - mean_squared_error: 100.4525
    Epoch 29/80
    414/414 [==============================] - 0s 26us/sample - loss: 99.7798 - mean_absolute_error: 7.3846 - mean_squared_error: 99.7798
    Epoch 30/80
    414/414 [==============================] - 0s 26us/sample - loss: 99.0432 - mean_absolute_error: 7.3550 - mean_squared_error: 99.0432
    Epoch 31/80
    414/414 [==============================] - 0s 29us/sample - loss: 98.2727 - mean_absolute_error: 7.2915 - mean_squared_error: 98.2727
    Epoch 32/80
    414/414 [==============================] - 0s 29us/sample - loss: 97.6225 - mean_absolute_error: 7.2673 - mean_squared_error: 97.6225
    Epoch 33/80
    414/414 [==============================] - 0s 29us/sample - loss: 96.9552 - mean_absolute_error: 7.2288 - mean_squared_error: 96.9552
    Epoch 34/80
    414/414 [==============================] - 0s 29us/sample - loss: 96.6670 - mean_absolute_error: 7.1995 - mean_squared_error: 96.6669
    Epoch 35/80
    414/414 [==============================] - 0s 31us/sample - loss: 95.8349 - mean_absolute_error: 7.1554 - mean_squared_error: 95.8349
    Epoch 36/80
    414/414 [==============================] - 0s 29us/sample - loss: 95.2536 - mean_absolute_error: 7.1659 - mean_squared_error: 95.2536
    Epoch 37/80
    414/414 [==============================] - 0s 29us/sample - loss: 94.7602 - mean_absolute_error: 7.1163 - mean_squared_error: 94.7602
    Epoch 38/80
    414/414 [==============================] - 0s 31us/sample - loss: 94.2784 - mean_absolute_error: 7.0691 - mean_squared_error: 94.2784
    Epoch 39/80
    414/414 [==============================] - 0s 31us/sample - loss: 93.8157 - mean_absolute_error: 7.0647 - mean_squared_error: 93.8157
    Epoch 40/80
    414/414 [==============================] - 0s 27us/sample - loss: 93.2790 - mean_absolute_error: 7.0255 - mean_squared_error: 93.2790
    Epoch 41/80
    414/414 [==============================] - 0s 31us/sample - loss: 92.8918 - mean_absolute_error: 6.9927 - mean_squared_error: 92.8918
    Epoch 42/80
    414/414 [==============================] - 0s 31us/sample - loss: 92.4318 - mean_absolute_error: 7.0090 - mean_squared_error: 92.4318
    Epoch 43/80
    414/414 [==============================] - 0s 36us/sample - loss: 91.9613 - mean_absolute_error: 6.9723 - mean_squared_error: 91.9613
    Epoch 44/80
    414/414 [==============================] - 0s 31us/sample - loss: 91.6935 - mean_absolute_error: 6.9076 - mean_squared_error: 91.6936
    Epoch 45/80
    414/414 [==============================] - 0s 31us/sample - loss: 91.2516 - mean_absolute_error: 6.9085 - mean_squared_error: 91.2516
    Epoch 46/80
    414/414 [==============================] - 0s 31us/sample - loss: 90.9863 - mean_absolute_error: 6.8781 - mean_squared_error: 90.9863
    Epoch 47/80
    414/414 [==============================] - 0s 29us/sample - loss: 90.6350 - mean_absolute_error: 6.8916 - mean_squared_error: 90.6350
    Epoch 48/80
    414/414 [==============================] - 0s 29us/sample - loss: 90.2415 - mean_absolute_error: 6.8412 - mean_squared_error: 90.2415
    Epoch 49/80
    414/414 [==============================] - 0s 29us/sample - loss: 89.9502 - mean_absolute_error: 6.8370 - mean_squared_error: 89.9502
    Epoch 50/80
    414/414 [==============================] - 0s 29us/sample - loss: 89.6989 - mean_absolute_error: 6.8029 - mean_squared_error: 89.6989
    Epoch 51/80
    414/414 [==============================] - 0s 36us/sample - loss: 89.4900 - mean_absolute_error: 6.7669 - mean_squared_error: 89.4900
    Epoch 52/80
    414/414 [==============================] - 0s 39us/sample - loss: 89.2736 - mean_absolute_error: 6.7756 - mean_squared_error: 89.2736
    Epoch 53/80
    414/414 [==============================] - 0s 39us/sample - loss: 88.8706 - mean_absolute_error: 6.7840 - mean_squared_error: 88.8706
    Epoch 54/80
    414/414 [==============================] - 0s 41us/sample - loss: 88.6695 - mean_absolute_error: 6.7358 - mean_squared_error: 88.6694
    Epoch 55/80
    414/414 [==============================] - 0s 34us/sample - loss: 88.6424 - mean_absolute_error: 6.7578 - mean_squared_error: 88.6424
    Epoch 56/80
    414/414 [==============================] - 0s 26us/sample - loss: 88.2962 - mean_absolute_error: 6.7128 - mean_squared_error: 88.2962
    Epoch 57/80
    414/414 [==============================] - 0s 29us/sample - loss: 87.9580 - mean_absolute_error: 6.7244 - mean_squared_error: 87.9580
    Epoch 58/80
    414/414 [==============================] - 0s 29us/sample - loss: 87.7580 - mean_absolute_error: 6.6752 - mean_squared_error: 87.7580
    Epoch 59/80
    414/414 [==============================] - 0s 29us/sample - loss: 87.6155 - mean_absolute_error: 6.6978 - mean_squared_error: 87.6154
    Epoch 60/80
    414/414 [==============================] - 0s 31us/sample - loss: 87.4292 - mean_absolute_error: 6.6308 - mean_squared_error: 87.4292
    Epoch 61/80
    414/414 [==============================] - 0s 36us/sample - loss: 87.1647 - mean_absolute_error: 6.6502 - mean_squared_error: 87.1647
    Epoch 62/80
    414/414 [==============================] - 0s 36us/sample - loss: 87.0084 - mean_absolute_error: 6.6526 - mean_squared_error: 87.0084
    Epoch 63/80
    414/414 [==============================] - 0s 36us/sample - loss: 86.8670 - mean_absolute_error: 6.6218 - mean_squared_error: 86.8670
    Epoch 64/80
    414/414 [==============================] - 0s 36us/sample - loss: 86.7010 - mean_absolute_error: 6.6240 - mean_squared_error: 86.7010
    Epoch 65/80
    414/414 [==============================] - 0s 39us/sample - loss: 86.6165 - mean_absolute_error: 6.6449 - mean_squared_error: 86.6165
    Epoch 66/80
    414/414 [==============================] - 0s 34us/sample - loss: 86.5450 - mean_absolute_error: 6.5980 - mean_squared_error: 86.5450
    Epoch 67/80
    414/414 [==============================] - 0s 31us/sample - loss: 86.2413 - mean_absolute_error: 6.5889 - mean_squared_error: 86.2413
    Epoch 68/80
    414/414 [==============================] - 0s 31us/sample - loss: 86.0718 - mean_absolute_error: 6.5681 - mean_squared_error: 86.0718
    Epoch 69/80
    414/414 [==============================] - 0s 31us/sample - loss: 86.0393 - mean_absolute_error: 6.5719 - mean_squared_error: 86.0393
    Epoch 70/80
    414/414 [==============================] - 0s 34us/sample - loss: 85.8159 - mean_absolute_error: 6.5754 - mean_squared_error: 85.8159
    Epoch 71/80
    414/414 [==============================] - 0s 31us/sample - loss: 85.8833 - mean_absolute_error: 6.5529 - mean_squared_error: 85.8833
    Epoch 72/80
    414/414 [==============================] - 0s 34us/sample - loss: 85.6118 - mean_absolute_error: 6.5541 - mean_squared_error: 85.6118
    Epoch 73/80
    414/414 [==============================] - 0s 34us/sample - loss: 85.5004 - mean_absolute_error: 6.5532 - mean_squared_error: 85.5004
    Epoch 74/80
    414/414 [==============================] - 0s 29us/sample - loss: 85.4184 - mean_absolute_error: 6.5230 - mean_squared_error: 85.4184
    Epoch 75/80
    414/414 [==============================] - 0s 34us/sample - loss: 85.2093 - mean_absolute_error: 6.5369 - mean_squared_error: 85.2093
    Epoch 76/80
    414/414 [==============================] - 0s 34us/sample - loss: 85.1984 - mean_absolute_error: 6.5266 - mean_squared_error: 85.1984
    Epoch 77/80
    414/414 [==============================] - 0s 29us/sample - loss: 85.1484 - mean_absolute_error: 6.5100 - mean_squared_error: 85.1484
    Epoch 78/80
    414/414 [==============================] - 0s 26us/sample - loss: 84.9540 - mean_absolute_error: 6.5362 - mean_squared_error: 84.9540
    Epoch 79/80
    414/414 [==============================] - 0s 26us/sample - loss: 84.8996 - mean_absolute_error: 6.5200 - mean_squared_error: 84.8996
    Epoch 80/80
    414/414 [==============================] - 0s 36us/sample - loss: 84.7856 - mean_absolute_error: 6.5134 - mean_squared_error: 84.7856
    




    <tensorflow.python.keras.callbacks.History at 0x12efa734e08>




```python
plt.plot(model.history.epoch, model.history.history['loss'])
```




    [<matplotlib.lines.Line2D at 0x12efaa53988>]

![img](output_26_1.png)

```python
mean_squared_error(y.values, np.array(model.predict(Xnn)).ravel())
```




    84.63828301314209




```python
r2_score(y.values, np.array(model.predict(Xnn)).ravel())
```




    0.5417261593449861



#### TensorFlow.Keras Deep Learning


```python
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

scaler = MinMaxScaler()
scaler.fit(X_train)
```




    MinMaxScaler(copy=True, feature_range=(0, 1))




```python
X_train_minmax = scaler.transform(X_train)
X_test_minmax = scaler.transform(X_test)
```


```python
X_train_minmax = sm.add_constant(X_train_minmax, prepend=False)
X_test_minmax = sm.add_constant(X_test_minmax, prepend=False)
```


```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation = 'relu', input_shape=[6]),
    tf.keras.layers.Dense(64, activation = 'relu', input_shape=[64]),
    tf.keras.layers.Dense(1)
])
optimizer = tf.keras.optimizers.Adam(0.01)
model.compile(loss = 'mean_squared_error',
              optimizer = optimizer,
              metrics = ['mean_squared_error'])
model.fit(X_train_minmax, y_train.values, epochs = 50)
```

    Train on 289 samples
    Epoch 1/50
    289/289 [==============================] - 0s 1ms/sample - loss: 1417.4486 - mean_squared_error: 1417.4485
    Epoch 2/50
    289/289 [==============================] - 0s 41us/sample - loss: 369.6510 - mean_squared_error: 369.6509
    Epoch 3/50
    289/289 [==============================] - 0s 45us/sample - loss: 273.5345 - mean_squared_error: 273.5345
    Epoch 4/50
    289/289 [==============================] - 0s 52us/sample - loss: 201.6184 - mean_squared_error: 201.6184
    Epoch 5/50
    289/289 [==============================] - 0s 48us/sample - loss: 136.8303 - mean_squared_error: 136.8303
    Epoch 6/50
    289/289 [==============================] - 0s 52us/sample - loss: 118.8896 - mean_squared_error: 118.8896
    Epoch 7/50
    289/289 [==============================] - 0s 52us/sample - loss: 106.5707 - mean_squared_error: 106.5707
    Epoch 8/50
    289/289 [==============================] - 0s 52us/sample - loss: 106.1355 - mean_squared_error: 106.1355
    Epoch 9/50
    289/289 [==============================] - 0s 48us/sample - loss: 98.2745 - mean_squared_error: 98.2745
    Epoch 10/50
    289/289 [==============================] - 0s 48us/sample - loss: 93.1996 - mean_squared_error: 93.1996
    Epoch 11/50
    289/289 [==============================] - 0s 52us/sample - loss: 90.7836 - mean_squared_error: 90.7836
    Epoch 12/50
    289/289 [==============================] - 0s 48us/sample - loss: 90.8065 - mean_squared_error: 90.8065
    Epoch 13/50
    289/289 [==============================] - 0s 48us/sample - loss: 89.8139 - mean_squared_error: 89.8139
    Epoch 14/50
    289/289 [==============================] - 0s 52us/sample - loss: 87.2888 - mean_squared_error: 87.2888
    Epoch 15/50
    289/289 [==============================] - 0s 52us/sample - loss: 84.6130 - mean_squared_error: 84.6130
    Epoch 16/50
    289/289 [==============================] - 0s 55us/sample - loss: 85.9935 - mean_squared_error: 85.9935
    Epoch 17/50
    289/289 [==============================] - 0s 59us/sample - loss: 84.2700 - mean_squared_error: 84.2700
    Epoch 18/50
    289/289 [==============================] - 0s 52us/sample - loss: 84.0978 - mean_squared_error: 84.0978
    Epoch 19/50
    289/289 [==============================] - ETA: 0s - loss: 33.6458 - mean_squared_error: 33.64 - 0s 55us/sample - loss: 83.8161 - mean_squared_error: 83.8161
    Epoch 20/50
    289/289 [==============================] - 0s 52us/sample - loss: 86.7847 - mean_squared_error: 86.7847
    Epoch 21/50
    289/289 [==============================] - 0s 55us/sample - loss: 82.5426 - mean_squared_error: 82.5426
    Epoch 22/50
    289/289 [==============================] - 0s 48us/sample - loss: 84.9819 - mean_squared_error: 84.9819
    Epoch 23/50
    289/289 [==============================] - 0s 52us/sample - loss: 80.7274 - mean_squared_error: 80.7274
    Epoch 24/50
    289/289 [==============================] - 0s 48us/sample - loss: 84.5170 - mean_squared_error: 84.5170
    Epoch 25/50
    289/289 [==============================] - 0s 59us/sample - loss: 80.4958 - mean_squared_error: 80.4958
    Epoch 26/50
    289/289 [==============================] - 0s 52us/sample - loss: 91.2925 - mean_squared_error: 91.2925
    Epoch 27/50
    289/289 [==============================] - 0s 48us/sample - loss: 84.0961 - mean_squared_error: 84.0961
    Epoch 28/50
    289/289 [==============================] - 0s 35us/sample - loss: 82.1278 - mean_squared_error: 82.1278
    Epoch 29/50
    289/289 [==============================] - 0s 48us/sample - loss: 80.0538 - mean_squared_error: 80.0538
    Epoch 30/50
    289/289 [==============================] - 0s 45us/sample - loss: 85.4851 - mean_squared_error: 85.4851
    Epoch 31/50
    289/289 [==============================] - 0s 41us/sample - loss: 82.5200 - mean_squared_error: 82.5200
    Epoch 32/50
    289/289 [==============================] - 0s 38us/sample - loss: 81.3487 - mean_squared_error: 81.3487
    Epoch 33/50
    289/289 [==============================] - 0s 38us/sample - loss: 85.8138 - mean_squared_error: 85.8138
    Epoch 34/50
    289/289 [==============================] - 0s 41us/sample - loss: 76.6028 - mean_squared_error: 76.6028
    Epoch 35/50
    289/289 [==============================] - 0s 38us/sample - loss: 78.6848 - mean_squared_error: 78.6848
    Epoch 36/50
    289/289 [==============================] - 0s 38us/sample - loss: 78.9861 - mean_squared_error: 78.9861
    Epoch 37/50
    289/289 [==============================] - 0s 38us/sample - loss: 75.7462 - mean_squared_error: 75.7462
    Epoch 38/50
    289/289 [==============================] - 0s 35us/sample - loss: 75.9305 - mean_squared_error: 75.9305
    Epoch 39/50
    289/289 [==============================] - 0s 41us/sample - loss: 83.8735 - mean_squared_error: 83.8735
    Epoch 40/50
    289/289 [==============================] - 0s 41us/sample - loss: 79.8874 - mean_squared_error: 79.8874
    Epoch 41/50
    289/289 [==============================] - 0s 48us/sample - loss: 76.3755 - mean_squared_error: 76.3755
    Epoch 42/50
    289/289 [==============================] - 0s 41us/sample - loss: 88.6059 - mean_squared_error: 88.6059
    Epoch 43/50
    289/289 [==============================] - 0s 41us/sample - loss: 75.5642 - mean_squared_error: 75.5642
    Epoch 44/50
    289/289 [==============================] - 0s 35us/sample - loss: 78.2108 - mean_squared_error: 78.2108
    Epoch 45/50
    289/289 [==============================] - 0s 38us/sample - loss: 76.0073 - mean_squared_error: 76.0073
    Epoch 46/50
    289/289 [==============================] - 0s 35us/sample - loss: 75.3020 - mean_squared_error: 75.3020
    Epoch 47/50
    289/289 [==============================] - 0s 35us/sample - loss: 71.8171 - mean_squared_error: 71.8171
    Epoch 48/50
    289/289 [==============================] - 0s 35us/sample - loss: 72.6085 - mean_squared_error: 72.6085
    Epoch 49/50
    289/289 [==============================] - 0s 31us/sample - loss: 74.4558 - mean_squared_error: 74.4558
    Epoch 50/50
    289/289 [==============================] - 0s 31us/sample - loss: 70.8407 - mean_squared_error: 70.8407
    




    <tensorflow.python.keras.callbacks.History at 0x12efc2d7688>




```python
plt.plot(model.history.epoch, model.history.history['loss'])
```




    [<matplotlib.lines.Line2D at 0x12efc7935c8>]


![img](output_34_1.png)


```python
y_train_pred = np.array(model.predict(X_train_minmax)).ravel()
y_test_pred = np.array(model.predict(X_test_minmax)).ravel()
```


```python
r2_score(y_train, y_train_pred)
```




    0.6360073978742543




```python
r2_score(y_test, y_test_pred)
```




    0.6584142389208574

