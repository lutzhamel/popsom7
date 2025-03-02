# Popsom7

### A User-friendly Implementation of Kohonen's Self-Organizing Maps

![](https://raw.githubusercontent.com/lutzhamel/popsom7/master/map.png)

An implementation of Kohonen's self-organizing maps with a number of distinguishing features:

1. Easy to use interfaces for building and evaluating self-organizing maps:
   * An interface that works the same on both the R and the Python platforms
   * An interface that is sklearn compatible, allowing you to leverage the power
     and convenience of the sklearn framework.

2. Automatic centroid detection and visualization using starbursts.

3. Two models of the data: (a) a self organizing map model, (b) a centroid based clustering model.

4. A number of easily accessible quality metrics for the self organizing map and the centroid based cluster model.

Other documents: 

* For a worked Python example check this [notebook on Kaggle](https://www.kaggle.com/code/lutzhamel/clustering-with-python-popsom7).

* For an example in R check this [notebook on Kaggle](https://www.kaggle.com/lutzhamel/customer-segmentation-with-soms).

* The documentation for the Python API inspired by the R implementation  can be found [here](https://github.com/lutzhamel/popsom7/blob/master/Python/man/maputils.md).

* The documentation for the sklearn compatible Python API  can be found [here](https://github.com/lutzhamel/popsom7/blob/master/Python/man/sklearnapi.md).

* For the R documentation as part of CRAN check [here](https://CRAN.R-project.org/package=popsom7).
