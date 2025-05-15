import sys; sys.path.append('../popsom7') # access to the Python code
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