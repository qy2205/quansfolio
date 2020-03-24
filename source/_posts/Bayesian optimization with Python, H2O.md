---
title: 'Bayesian optimization with Python, H2O'
summary: This blog used Python and H2O to implement Bayesian Optimization method for parameters tunning.
date: 2020-01-10 21:51:19
categories:
  - data science
tags:
  - optimization
  - H2O
---

```python
# install and import packages (make sure you have installed Java)
# !pip install bayesian-optimization
# !pip install h2o
import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from bayes_opt import BayesianOptimization
```


```python
# start h2o
h2o.init()
h2o.remove_all()
```

    Checking whether there is an H2O instance running at http://localhost:54321 ..... not found.
    Attempting to start a local H2O server...
    ; Java HotSpot(TM) 64-Bit Server VM (build 13.0.1+9, mixed mode, sharing)
      Starting server from X:\Anaconda3\lib\site-packages\h2o\backend\bin\h2o.jar
      Ice root: C:\Users\quany\AppData\Local\Temp\tmpx6umusj0
      JVM stdout: C:\Users\quany\AppData\Local\Temp\tmpx6umusj0\h2o_quany_started_from_python.out
      JVM stderr: C:\Users\quany\AppData\Local\Temp\tmpx6umusj0\h2o_quany_started_from_python.err
      Server is running at http://127.0.0.1:54321
    Connecting to H2O server at http://127.0.0.1:54321 ... successful.
    


<div style="overflow:auto"><table style="width:50%"><tr><td>H2O cluster uptime:</td>
<td>02 secs</td></tr>
<tr><td>H2O cluster timezone:</td>
<td>America/New_York</td></tr>
<tr><td>H2O data parsing timezone:</td>
<td>UTC</td></tr>
<tr><td>H2O cluster version:</td>
<td>3.28.0.1</td></tr>
<tr><td>H2O cluster version age:</td>
<td>25 days </td></tr>
<tr><td>H2O cluster name:</td>
<td>H2O_from_python_quany_qvstr7</td></tr>
<tr><td>H2O cluster total nodes:</td>
<td>1</td></tr>
<tr><td>H2O cluster free memory:</td>
<td>3.959 Gb</td></tr>
<tr><td>H2O cluster total cores:</td>
<td>16</td></tr>
<tr><td>H2O cluster allowed cores:</td>
<td>16</td></tr>
<tr><td>H2O cluster status:</td>
<td>accepting new members, healthy</td></tr>
<tr><td>H2O connection url:</td>
<td>http://127.0.0.1:54321</td></tr>
<tr><td>H2O connection proxy:</td>
<td>{'http': None, 'https': None}</td></tr>
<tr><td>H2O internal security:</td>
<td>False</td></tr>
<tr><td>H2O API Extensions:</td>
<td>Amazon S3, Algos, AutoML, Core V3, TargetEncoder, Core V4</td></tr>
<tr><td>Python version:</td>
<td>3.7.4 final</td></tr></table></div>



```python
# load dataset
data = h2o.upload_file("winequality-red.csv")
```

    Parse progress: |█████████████████████████████████████████████████████████| 100%

```python
# train test split
train, test = data.split_frame(ratios = [0.7], destination_frames = ['train', 'test'])
```

Suppose we want to tune the following parameters:  
max_depth, ntrees, min_rows, learn_rate, sample_rate, col_sample_rate  

BayesianOptimization packages need two parts for achieving that  
(1) The Target function  
(2) The boundary of parameters  


```python
# The Target function
def GBDT_h2o(max_depth, ntrees, min_rows, learn_rate, sample_rate, \
             data = train, xcols = train.columns[:-1], ycol = 'quality'):
    params = {'max_depth': int(max_depth),
              'ntrees': int(ntrees),
              'min_rows': int(min_rows),
              'learn_rate': learn_rate,
              'sample_rate': sample_rate}
    # not specify nfolds = 5
    model = H2OGradientBoostingEstimator(**params)
    model.train(x = xcols, y = ycol, training_frame = data, validation_frame = test)
#     train_rmse = - model.rmse()
    test_rmse = - model.rmse(valid=True)
    return test_rmse

# Optimization boundaries
bounds = {'max_depth':(3, 8),
          'ntrees': (300, 800),
          'min_rows': (5, 10),
          'learn_rate': (0.01, 0.05),
          'sample_rate': (0.8, 1)}
```


```python
# run Bayesian Optimization
optimizer = BayesianOptimization(
    f = GBDT_h2o,
    pbounds = bounds,
    random_state = 2020)
optimizer.maximize(init_points = 5, n_iter = 20)
```

    |   iter    |  target   | learn_... | max_depth | min_rows  |  ntrees   | sample... |
    -------------------------------------------------------------------------------------
    gbm Model Build progress: |███████████████████████████████████████████████| 100%
    |  1        | -0.5718   |  0.04945  |  7.367    |  7.549    |  435.9    |  0.8674   |
    gbm Model Build progress: |███████████████████████████████████████████████| 100%
    |  2        | -0.5827   |  0.01868  |  4.382    |  6.717    |  731.1    |  0.8313   |
    gbm Model Build progress: |███████████████████████████████████████████████| 100%
    |  3        | -0.5708   |  0.01564  |  6.785    |  8.682    |  477.8    |  0.8682   |
    =====================================================================================

```python
optimizer.max
```

    {'target': -0.5632478247168863,
     'params': {'learn_rate': 0.01,
      'max_depth': 8.0,
      'min_rows': 5.000000007839401,
      'ntrees': 800.0,
      'sample_rate': 0.8000000148665619}}

```python
# train model with the best parameters
best_param = {'learn_rate': 0.01,
             'max_depth': 8,
             'min_rows': 5,
             'ntrees': 800,
             'sample_rate': 0.8}
# nfolds = 5
model = H2OGradientBoostingEstimator(**best_param)
model.train(x = train.columns[:-1], y = 'quality', \
            training_frame = train, validation_frame = test)
```

    gbm Model Build progress: |███████████████████████████████████████████████| 100%

```python
print("Train RMSE = {0}".format(model.rmse()))
print("Test RMSE = {0}".format(model.rmse(valid=True)))
```

    Train RMSE = 0.16036605054753
    Test RMSE = 0.5652282702693677
    
```python
# predict the test value
model.predict(test)
```

    gbm prediction progress: |████████████████████████████████████████████████| 100%
    

<table>
<thead>
<tr><th style="text-align: right;">  predict</th></tr>
</thead>
<tbody>
<tr><td style="text-align: right;">  4.97194</td></tr>
<tr><td style="text-align: right;">  5.12012</td></tr>
<tr><td style="text-align: right;">  4.90172</td></tr>
<tr><td style="text-align: right;">  5.42199</td></tr>
<tr><td style="text-align: right;">  5.02189</td></tr>
<tr><td style="text-align: right;">  5.24796</td></tr>
<tr><td style="text-align: right;">  5.24851</td></tr>
<tr><td style="text-align: right;">  5.23156</td></tr>
<tr><td style="text-align: right;">  5.72173</td></tr>
<tr><td style="text-align: right;">  5.31643</td></tr>
</tbody>
</table>