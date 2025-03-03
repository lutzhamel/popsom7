# A Fast, User-friendly Implementation of Self-Organizing Maps


![](https://raw.githubusercontent.com/lutzhamel/popsom7/master/map.png)

## Overview

An implementation of self-organizing maps (SOMs) with a number of distinguishing features:

1. Support for both Python and R.

1. Easy to use interfaces for building and evaluating self-organizing maps:
   * An interface that works the same on both the R and the Python platforms
   * An interface that is **sklearn compatible**, allowing you to leverage the power
     and convenience of the sklearn framework in Python.

1. Automatic centroid detection and visualization using starbursts.

1. Two models of the data: (a) a self organizing map model, (b) a centroid based clustering model.

1. A number of easily accessible quality metrics.

1. An implementation of the training algorithm based on tensor algebra.

## Example Code

Here is a simple example written in the sklearn compatible API,
```python
   from popsom7.sklearnapi import SOM
   import pandas as pd
   from sklearn import datasets

   iris = datasets.load_iris()
   X = pd.DataFrame(iris.data, columns=iris.feature_names)
   y = pd.DataFrame(iris.target_names[iris.target],columns=['species'])

   # Create and fit the SOM model
   som = SOM(xdim=20, ydim=15, train=100000, seed=42).fit(X, y)

   # View a summary of the SOM
   som.summary()

   # Display the starburst (heat map) representation
   som.starburst()

   # Display feature significance and a marginal plots
   print(som.significance())
   som.marginal(2)
```

## Documentation

* The documentation for the [sklearn compatible Python API](https://lutzhamel.github.io/popsom7/Python/man/sklearnapi.html).

* The documentation for the [Python API based on the R implementation](https://lutzhamel.github.io/popsom7/Python/man/maputils.html).

* The [R documentation as part of CRAN](https://cran.r-project.org/web/packages/popsom7/popsom7.pdf).


## Worked Example Jupyter Notebooks

* [Python notebook](https://www.kaggle.com/code/lutzhamel/clustering-with-python-popsom7).

* [R notebook on Kaggle](https://www.kaggle.com/lutzhamel/customer-segmentation-with-soms).

## Github 

[Project Page](https://github.com/lutzhamel/popsom7)
