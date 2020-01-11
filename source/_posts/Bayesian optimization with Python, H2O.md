---
title: 'Bayesian optimization with Python, H2O'
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
    gbm Model Build progress: |███████████████████████████████████████████████| 100%
    |  4        | -0.5796   |  0.03667  |  4.086    |  7.807    |  362.1    |  0.8639   |
    gbm Model Build progress: |███████████████████████████████████████████████| 100%
    |  5        | -0.5954   |  0.04813  |  3.687    |  7.847    |  787.8    |  0.9007   |
    gbm Model Build progress: |███████████████████████████████████████████████| 100%
    |  6        | -0.5944   |  0.04496  |  3.046    |  5.01     |  548.2    |  0.8695   |
    gbm Model Build progress: |███████████████████████████████████████████████| 100%
    |  7        | -0.5785   |  0.01366  |  7.912    |  9.981    |  300.5    |  0.9784   |
    gbm Model Build progress: |███████████████████████████████████████████████| 100%
    |  8        | -0.5798   |  0.04055  |  7.891    |  9.995    |  647.1    |  0.8996   |
    gbm Model Build progress: |███████████████████████████████████████████████| 100%
    |  9        | -0.5913   |  0.03792  |  3.181    |  9.985    |  449.0    |  0.9515   |
    gbm Model Build progress: |███████████████████████████████████████████████| 100%
    |  10       | -0.5857   |  0.05     |  8.0      |  10.0     |  370.5    |  1.0      |
    gbm Model Build progress: |███████████████████████████████████████████████| 100%
    |  11       | -0.5668   |  0.04026  |  7.982    |  5.0      |  321.7    |  0.8257   |
    gbm Model Build progress: |███████████████████████████████████████████████| 100%
    |  12       | -0.5669   |  0.03006  |  7.983    |  5.055    |  680.2    |  0.939    |
    gbm Model Build progress: |███████████████████████████████████████████████| 100%
    |  13       | -0.5735   |  0.05     |  7.919    |  5.085    |  479.0    |  0.9933   |
    gbm Model Build progress: |███████████████████████████████████████████████| 100%
    |  14       | -0.5767   |  0.03448  |  7.996    |  9.901    |  529.1    |  0.9322   |
    gbm Model Build progress: |███████████████████████████████████████████████| 100%
    |  15       | -0.5745   |  0.04907  |  7.985    |  5.332    |  596.2    |  0.8916   |
    gbm Model Build progress: |███████████████████████████████████████████████| 100%
    |  16       | -0.5941   |  0.0368   |  3.138    |  5.014    |  300.6    |  0.9859   |
    gbm Model Build progress: |███████████████████████████████████████████████| 100%
    |  17       | -0.5863   |  0.04908  |  3.003    |  9.974    |  691.5    |  0.8307   |
    gbm Model Build progress: |███████████████████████████████████████████████| 100%
    |  18       | -0.5888   |  0.02112  |  3.078    |  9.961    |  329.1    |  0.9597   |
    gbm Model Build progress: |███████████████████████████████████████████████| 100%
    |  19       | -0.5632   |  0.01     |  8.0      |  5.0      |  800.0    |  0.8      |
    gbm Model Build progress: |███████████████████████████████████████████████| 100%
    |  20       | -0.5912   |  0.04766  |  7.951    |  9.958    |  714.0    |  0.8181   |
    gbm Model Build progress: |███████████████████████████████████████████████| 100%
    |  21       | -0.5948   |  0.03614  |  3.032    |  5.182    |  628.9    |  0.9698   |
    gbm Model Build progress: |███████████████████████████████████████████████| 100%
    |  22       | -0.5787   |  0.03803  |  7.981    |  9.942    |  569.8    |  0.9471   |
    gbm Model Build progress: |███████████████████████████████████████████████| 100%
    |  23       | -0.5736   |  0.0495   |  7.91     |  5.046    |  767.2    |  0.9536   |
    gbm Model Build progress: |███████████████████████████████████████████████| 100%
    |  24       | -0.5694   |  0.01296  |  7.796    |  5.001    |  405.8    |  0.8278   |
    gbm Model Build progress: |███████████████████████████████████████████████| 100%
    |  25       | -0.5697   |  0.04384  |  7.976    |  5.199    |  347.3    |  0.9118   |
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