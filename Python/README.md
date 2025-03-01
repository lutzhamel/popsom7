# popsom7

**popsom7** is a Python package that provides a collection of routines for constructing and evaluating self-organizing maps. It also has a sklearn compatible
interface.  The functionality includes:

- Building a SOM with customizable dimensions and training parameters.
- Generating summary statistics for the SOM.
- Visualizing the map (e.g., starburst/heat map display).
- Assessing feature significance and marginal density plots among other quality measures.
- Predicting classifications and positions for new data points.
- Ease of use and easily accessible data models are one of the  hallmarks of this module.

## Installation

You can install popsom7 via pip:

```bash
pip install popsom7
```

## Usage
Below is a quick example using the popsom `sklearnapi` interface.  Popsom also supports
an interface similar to the popsom R release.  For more details please see the project [homepage](https://github.com/lutzhamel/popsom7) 

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

