import sys; sys.path.append('../popsom7') # access to the Python code
import unittest
import pandas as pd
import maputils
from sklearn import datasets

iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.DataFrame(iris.target_names[iris.target],columns=['species'])

m = maputils.map_build(X, 
                       labels=y, 
                       xdim=20, 
                       ydim=15, 
                       train=100000, 
                       seed=42)

maputils.map_summary(m)

class TestMaputils(unittest.TestCase):
    def test_build(self):
      # check the map built above
      self.assertTrue(m['convergence'] > 0.8)
      self.assertTrue(len(m['unique_centroids'])<9)

    def test_fitted(self):
      v = maputils.map_fitted(m)
      # note: we cannot test for specific values because
      # different random number generators/OSs will generate
      # different behavior.  Here we make sure that at
      # least the returned structure makes sense.
      self.assertTrue(len(v) == X.shape[0])

    def test_marginal(self):
      # nothing to compare to but just run
      # the code and make sure it doesn't crash
      self.assertIsNot(m, None)
      maputils.map_marginal(m,1)

    def test_position(self):
      p = maputils.map_position(m,X)
      # note: we cannot test for specific values because
      # different random number generators/OSs will generate
      # different behavior.  Here we make sure that at
      # least the returned structure makes sense.
      self.assertTrue(p.shape[0]== X.shape[0])
      self.assertEqual(p.shape[1], 2)

    def test_predict(self):
      p  = maputils.map_predict(m,X)
      # spot check predictions with high confidence
      # for correct prediction
      for i in range(150):
        if p.iloc[i,1] > 0.9:
          self.assertEqual(str(p.iloc[i,0]),str(y.iloc[i,0]))
          break
        
    def test_significance(self):
      s = maputils.map_significance(m)
      # spot check the significance vector for most
      # significant feature
      self.assertTrue(0.4 < s.iloc[2] and s.iloc[2]< 0.7)

    def test_starburst(self):
      # nothing to compare to but just run
      # the code and make sure it doesn't crash
      self.assertIsNot(m, None)
      maputils.map_starburst(m)

    def test_summary(self):
      # run summary and pull out convergence from the report tables
      s = maputils.map_summary(m,verb=False)
      # grab the convergence field
      conv = s['quality_assessments']['convergence'].iloc[0]
      self.assertTrue(conv > 0.8)

if __name__ == '__main__':
    unittest.main()
