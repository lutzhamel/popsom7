import unittest
import pandas as pd
from popsom7 import maputils

class TestMaputils(unittest.TestCase):
    def test_build(self):
      # Load the iris dataset
      iris = pd.read_csv('iris.csv')
      print(iris.head())

      # Separate the features and the labels
      df = iris.drop(columns=['id','Species'])
      labels = iris[['Species']]

      # Build the map
      som_map = maputils.map_build(df, labels=labels, xdim=15, ydim=10, alpha=0.3, train=1000, normalize=False, seed=42)

      # check the map 
      self.assertEqual(som_map['convergence'] > 0.8,True))
      self.assertEqual(len(som_map['unique_centroids'])<9,True)

if __name__ == '__main__':
    unittest.main()
