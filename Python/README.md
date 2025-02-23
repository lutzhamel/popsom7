# popsom7

**popsom7** is a Python package that provides a collection of routines for constructing and evaluating self-organizing maps. The functionality includes:

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
Below is a quick example using the popsom `maputils` interface.  Popsom also supports
a sklearn compatible interface.  For more details please see the project [homepage](https://github.com/lutzhamel/popsom7) 

```python
import pandas as pd
from popsom7 import maputils
from sklearn import datasets

iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.DataFrame(iris.target_names[iris.target],columns=['species'])

# Build the SOM
som_map = maputils.map_build(X, labels=y, xdim=15, ydim=10, alpha=0.3, train=10000, seed=42)

# Print a summary of the map
maputils.map_summary(som_map)

# Display the starburst (heat map) visualization
maputils.map_starburst(som_map)
```

