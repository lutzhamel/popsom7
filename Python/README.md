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
Below is a quick example:

```python
import pandas as pd
from popsom7 import maputils

# Load your data (for example, a CSV file)
data = pd.read_csv('iris.csv')

# Optionally, separate out labels if available
labels = data[['Species']]
data = data.drop(columns=['Species'])

# Build the SOM 
som_map = maputils.map_build(data, labels=labels, xdim=10, ydim=5, alpha=0.3, train=1000, normalize=True, seed=42)

# Print a summary of the map
maputils.map_summary(som_map)

# Display the starburst (heat map) visualization
maputils.map_starburst(som_map)
```

## Documentation

Documentation of the API can be found [here](https://github.com/lutzhamel/popsom7/blob/master/Python/man/documentation.md)
