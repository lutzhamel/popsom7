# Popsom7

## Overview

A fast, user-friendly implementation of self-organizing maps (SOMs) with a number of distinguishing features:

1. Support for both Python and R.

1. Easy to use interfaces for building and evaluating self-organizing maps:
   * An interface that works the same on both the R and the Python platforms
   * An interface that is **sklearn compatible**, allowing you to leverage the power
     and convenience of the sklearn framework in Python.

1. Automatic centroid detection and visualization using starbursts.

1. Two models of the data: (a) a self organizing map model, (b) a centroid based clustering model.

1. A number of easily accessible quality metrics.

1. An implementation of the training algorithm based on tensor algebra.


## Installation

You can install popsom7 via pip:

```bash
pip install popsom7
```

## Usage
Below is a quick example using the popsom `sklearnapi` interface.   

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
```

Here is the same example written in the API based on the R API.
```python
from popsom7.maputils import map_build, map_summary, map_starburst
import pandas as pd
from sklearn import datasets   

iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.DataFrame(iris.target_names[iris.target],columns=['species'])

# Build the map
som_map = map_build(X, labels=y, xdim=20, ydim=15, train=100000, seed=42)

# View a summary of the SOM
map_summary(som_map)

# Display the starburst (heat map) representation
map_starburst(som_map)
```

For more details please see the [project homepage](https://github.com/lutzhamel/popsom7)