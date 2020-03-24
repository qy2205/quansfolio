---
title: 'Regression: TensorFlow vs sklearn vs StatsModels'
categories:
  - data science
summary: This blog compared three different packages' performance on linear regression method. Finally, we believe Statsmodels is the best option, result of tensorflow is not stable.
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
    <tensorflow.python.keras.callbacks.History at 0x12efa734e08>


```python
plt.plot(model.history.epoch, model.history.history['loss'])
```

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
    <tensorflow.python.keras.callbacks.History at 0x12efc2d7688>

```python
plt.plot(model.history.epoch, model.history.history['loss'])
```

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

