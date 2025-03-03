"""
map_utils.py
(Translated from maputils.R)
(c) University of Rhode Island
            Lutz Hamel, Benjamin Ott, Greg Breard,
            Robert Tatoian, Vishakh Gopu, Michael Eiger

This file constitutes a set of routines which are useful in constructing
and evaluating self-organizing maps (SOMs).

The main utilities available in this file are:
    map_build -------- constructs a map
    map_summary ------ compute a summary object
    map_convergence -- details of the convergence index of a map
    map_starburst ---- displays the starburst representation of the SOM model,
                         the centers of starbursts are the centers of clusters
    map_fitted ------- returns a vector of labels assigned to the observations
    map_predict ------ returns classification labels for points in DF
    map_position ----- return the position of points on the map
    map_significance - graphically reports the significance of each feature with
                         respect to the self-organizing map model
    map_marginal ----- displays a density plot of a training dataframe dimension
                         overlayed with the neuron density for that same dimension or index.
    
License: GNU License
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import ttest_ind, ks_2samp, f
from scipy.ndimage import gaussian_filter

#---------------------------
# Helper class for coordinates
#---------------------------
class Coord:
    def __init__(self, x=-1, y=-1):
        self.x = x
        self.y = y
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    def __repr__(self):
        return f"Coord(x={self.x}, y={self.y})"

#---------------------------
# map_build -- construct a SOM map
#---------------------------
def map_build(data, labels=None, xdim=10, ydim=5, alpha=0.3, train=1000, normalize=False, seed=None, minimal=False):
    """
    map_build -- construct a SOM, returns an object of type 'map'
    
    parameters:
      - data: a DataFrame where each row contains an unlabeled training instance
      - labels: a vector or DataFrame with one label for each observation in data
      - xdim, ydim: the dimensions of the map
      - alpha: the learning rate (0 < alpha <= 1)
      - train: number of training iterations
      - normalize: whether to normalize the input data by row
      - seed: a seed value for repeatability
      - minimal: if True, only the trained neurons are returned.
    value:
      - a dict representing a map object
    """
    # Parameter checking
    if alpha <= 0 or alpha > 1:
        raise ValueError("invalid value for alpha")
    if xdim < 5 or ydim < 5:
        raise ValueError("map is too small.")
    if not isinstance(data, pd.DataFrame):
        raise ValueError("training data has to be a data frame")
    if not all(np.issubdtype(dt, np.number) for dt in data.dtypes):
        raise ValueError("only numeric data can be used for training")
    if labels is not None and not isinstance(labels, pd.DataFrame):
        labels = pd.DataFrame(labels)
    if normalize:
        data = map_normalize(data)
    if seed is not None:
        if seed <= 0:
            raise ValueError("seed value has to be a positive integer value")
        if not test_integer(seed):
            raise ValueError("seed value has to be a positive integer value")
        np.random.seed(int(seed))
    if not test_integer(train):
        raise ValueError("train value has to be a positive integer value")
    
    # Train the neural network using vsom_r
    # from datetime import datetime
    # now = datetime.now()
    neurons_array = vsom(data, xdim, ydim, alpha, train)
    # print(datetime.now()-now)
    neurons = pd.DataFrame(neurons_array, columns=data.columns)
    
    # Construct the map object as a dictionary
    m = {
        'data': data,
        'labels': labels,
        'xdim': xdim,
        'ydim': ydim,
        'alpha': alpha,
        'train': train,
        'normalize': normalize,
        'seed': seed,
        'neurons': neurons
    }
    if minimal:
        m['class'] = 'map.minimal'
        return m
    else:
        m['class'] = 'map'
    
    # NOTE: do not change the order of the following computations
    m['heat'] = compute_heat(m)
    m['fitted_obs'] = map_fitted_obs(m)
    m['centroids'] = compute_centroids(m)
    m['unique_centroids'] = get_unique_centroids(m)
    m['centroid_labels'] = majority_labels(m)
    m['label_to_centroid'] = compute_label_to_centroid(m)
    m['centroid_obs'] = compute_centroid_obs(m)
    m['convergence'] = map_convergence(m, verb=False)
    m['wcss'] = compute_wcss(m)
    m['bcss'] = compute_bcss(m)
    
    return m

#---------------------------
# map_summary -- print/return a summary object
#---------------------------
def map_summary(m, verb=True):
    """
    compute a summary object of the map.
    
    parameters:
      - m: a map object (dict)
      - verb: if True, prints the summary.
    return:
      - a dict containing summary information.
    """
    if m.get('class') not in ['map']:
        raise ValueError("first argument is not a map object.")
    value = {}
    header = ["xdim", "ydim", "alpha", "train", "normalize", "seed", "instances", "columns"]
    v = [m['xdim'], 
         m['ydim'], 
         m['alpha'], 
         m['train'],
         "TRUE" if m['normalize'] else "FALSE",
         "NULL" if m['seed'] is None else m['seed'],
         m['data'].shape[0], 
         m['data'].shape[1]]
    df1 = pd.DataFrame([v], columns=header, index=[" "])
    value['training_parameters'] = df1

    header2 = ["convergence", "separation", "clusters"]
    separation = 1.0 - m['wcss'] / m['bcss'] if m['bcss'] != 0 else None
    v2 = [m['convergence'], 
          separation, 
          len(m['unique_centroids'])]
    df2 = pd.DataFrame([v2], columns=header2, index=[" "])
    value['quality_assessments'] = df2

    if verb:
        print("\nTraining Parameters:")
        print(df1)
        print("\nQuality Assessments:")
        print(df2.round(2))
        print("\n")
    else:
        return value

#---------------------------
# map_starburst -- display the starburst representation (i.e. heat map)
#---------------------------
def map_starburst(m):
    """
    displays the starburst representation of clusters.
    parameters:
      - m: a map object.
    """
    if m.get('class') not in ['map']:
        raise ValueError("first argument is not a map object.")
    plot_heat(m)

#---------------------------
# map_significance -- compute and (optionally) plot feature significance
#---------------------------
def map_significance(m, graphics=False, feature_labels=True):
    """
    compute the relative significance of each feature 
    based on variance and plot it.
    
    parameters:
      - m: a map object.
      - graphics: if True, produces a plot.
      - feature_labels: if True, uses feature names in the plot.
    return:
      - a vector (list) containing the significance for each feature.
    """
    if m.get('class') not in ['map']:
        raise ValueError("first argument is not a map object.")
    
    data_df = m['data']
    nfeatures = data_df.shape[1]

    # compute probability of each feature based on variance
    var_v = np.ones(nfeatures)
    for i in range(nfeatures):
        var_v[i] = data_df.iloc[:, i].var()
    var_sum = var_v.sum()
    prob_v = var_v / var_sum if var_sum != 0 else var_v

    if graphics:
        y_max = prob_v.max()
        plt.figure()
        plt.xlim(1, nfeatures)
        plt.ylim(0, y_max)
        plt.xlabel("Features")
        plt.ylabel("Significance")
        xticks = np.arange(1, nfeatures + 1, 1)
        yticks = np.linspace(0, y_max, 5)
        if feature_labels:
            xlabels = data_df.columns.tolist()
        else:
            xlabels = list(range(1, nfeatures + 1))
        plt.xticks(xticks, xlabels)
        plt.yticks(yticks, [f"{val:.2f}" for val in yticks])
        for i in range(nfeatures):
            plt.plot([i + 1, i + 1], [0, prob_v[i]], marker="o")
        plt.show()
        return None
    else:
        return pd.Series(prob_v, index=data_df.columns)

#---------------------------
# map_marginal -- plot marginal density for a chosen dimension
#---------------------------
def map_marginal(m, marginal):
    """
    a plot that shows the marginal probability distribution of the neurons and data.
    
    parameters:
      - m: a map object.
      - marginal: the name of a training data frame dimension or index.
    """
    if m.get('class') not in ['map']:
        raise ValueError("first argument is not a map object.")
    
    if isinstance(marginal, int) or isinstance(marginal, np.integer):
        train = pd.DataFrame({'points': m['data'].iloc[:, marginal]})
        neurons = pd.DataFrame({'points': m['neurons'].iloc[:, marginal]})
        train['legend'] = 'training data'
        neurons['legend'] = 'neurons'
        hist = pd.concat([train, neurons])
        ax = sns.kdeplot(data=hist, x="points", hue="legend", fill=True, alpha=0.2)
        ax.set_xlabel(m['data'].columns[marginal])
        plt.show()
    elif isinstance(marginal, str) and marginal in m['data'].columns:
        train = pd.DataFrame({'points': m['data'][marginal]})
        neurons = pd.DataFrame({'points': m['neurons'][marginal]})
        train['legend'] = 'training data'
        neurons['legend'] = 'neurons'
        hist = pd.concat([train, neurons])
        ax = sns.kdeplot(data=hist, x="points", hue="legend", fill=True, alpha=0.2)
        ax.set_xlabel(marginal)
        plt.show()
    else:
        raise ValueError("second argument is not a data dimension or index")

#---------------------------
# map_fitted -- return vector of labels assigned to observations
#---------------------------
def map_fitted(m):
   """
   return a vector of labels assigned to the training observations.

   parameters:
      - m: a map object.
   value:
      - a vector of labels
   """
   if m.get('class') not in ['map']:
      raise ValueError("first argument is not a map object.")
   nobs = len(m['fitted_obs'])
   labels_out = []
   for i in range(nobs):
      nix = m['fitted_obs'][i]
      coord_obj = coordinate(m, nix)
      x = coord_obj.x
      y = coord_obj.y
      centroid = m['centroids'][x - 1][y - 1]
      cx = centroid.x
      cy = centroid.y
      l = m['centroid_labels'][cx - 1][cy - 1]
      labels_out.append(l)
   return labels_out

#---------------------------
# map_predict -- return classification labels and confidence for new points
#---------------------------
def map_predict(m, points):
   """
   returns classification labels for points.
   
   parameters:
   - m: a map object.
   - points: a DataFrame or 1D array-like of points to be classified.
   return:
   - a DataFrame with columns "Label" and "Confidence".
   """
   if m.get('class') not in ['map']:
      raise ValueError("first argument is not a map object.")
   def predict_point(x):
      if not isinstance(x, (list, np.ndarray, pd.Series)):
         raise ValueError("argument has to be a vector.")
      if len(x) != m['data'].shape[1]:
         raise ValueError("predict vector dimensionality is incompatible")
      if m['normalize']:
         x = np.array(map_normalize(x))
      nix = best_match(m, x)
      coord_obj = coordinate(m, nix)
      ix = coord_obj.x
      iy = coord_obj.y
      c_xy = m['centroids'][ix - 1][iy - 1]
      c_nix = rowix(m, c_xy)
      label = m['centroid_labels'][c_xy.x - 1][c_xy.y - 1]
      c_ix = find_centroidix(m, c_xy)
      # Compute confidence: distance from x to centroid
      vectors = np.vstack([m['neurons'].iloc[c_nix - 1].values, np.array(x)])
      d_matrix = squareform(pdist(vectors))
      x_to_c_distance = np.max(d_matrix[0, :])
      # Now compute the cluster radius using training data
      c_ix = find_centroidix(m, c_xy)
      obs_indices = m['centroid_obs'][c_ix - 1]
      vectors = m['neurons'].iloc[[c_nix - 1]].values
      for obs_ix in obs_indices:
         vectors = np.vstack([vectors, m['data'].iloc[obs_ix - 1].values])
      if len(vectors) > 1:
         d_matrix = squareform(pdist(vectors))
         max_o_to_c_distance = np.max(d_matrix[0, :])
      else:
         max_o_to_c_distance = 0
      max_o_to_c_distance += 0.05 * max_o_to_c_distance
      conf = 1.0 - (x_to_c_distance / max_o_to_c_distance) if max_o_to_c_distance != 0 else 0
      return [label, conf]
   if isinstance(points, (list, np.ndarray, pd.Series)):
      points = pd.DataFrame([points], columns=m['data'].columns)
   results = points.apply(lambda row: pd.Series(predict_point(row.values)), axis=1)
   results.columns = ["Label", "Confidence"]
   return results

#---------------------------
# map_position -- return x-y coordinates on the map for new points
#---------------------------
def map_position(m, points):
   """
   return the position of points on the map.
   
   parameters:
   - m: a map object.
   - points: a DataFrame or vector of points.
   return:
   - a DataFrame with columns "x-dim" and "y-dim".
   """
   if m.get('class') not in ['map']:
      raise ValueError("first argument is not a map object.")
   def position_point(x):
      if not isinstance(x, (list, np.ndarray, pd.Series)):
         raise ValueError("argument has to be a vector.")
      if len(x) != m['data'].shape[1]:
         raise ValueError("vector dimensionality is incompatible")
      if m['normalize']:
         x = np.array(map_normalize(x))
      nix = best_match(m, x)
      coord_obj = coordinate(m, nix)
      return [coord_obj.x, coord_obj.y]
   if isinstance(points, (list, np.ndarray, pd.Series)):
      points = pd.DataFrame([points], columns=m['data'].columns)
   results = points.apply(lambda row: pd.Series(position_point(row.values)), axis=1)
   results.columns = ["x-dim", "y-dim"]
   return results

#---------------------------
# map_convergence -- compute the convergence index (embed and topo components)
#---------------------------
def map_convergence(m, conf_int=0.95, k=50, verb=True, ks=True):
   """
   details of the convergence index of a map.
   
   parameters:
   - m: a map object.
   - conf_int: confidence interval (default 95%)
   - k: number of samples for topographic accuracy estimation.
   - verb: if True, returns the two components separately.
   - ks: if True, uses the KS-test; if False, uses variance and means tests.
   return:
   - if verb: dict with keys 'embed' and 'topo'; otherwise, a single combined value.
   """
   if m.get('class') not in ['map']:
      raise ValueError("first argument is not a map object.")
   if ks:
      embed = map_embed_ks(m, conf_int, verb=False)
   else:
      embed = map_embed_vm(m, conf_int, verb=False)
   topo = map_topo(m, k, conf_int, verb=False, interval=False)
   if verb:
      return {'embed': embed, 'topo': topo}
   else:
      return 0.5 * embed + 0.5 * topo

#---------------------------
# Local helper functions
#---------------------------
def test_integer(x):
    return isinstance(x, int) or (isinstance(x, float) and x.is_integer())

def find_centroidix(m, cd):
    for i, centroid in enumerate(m['unique_centroids'], start=1):
        if cd.x == centroid.x and cd.y == centroid.y:
            return i
    raise ValueError("coordinate not a centroid")

def compute_centroid_obs(m):
    centroid_obs = [[] for _ in range(len(m['unique_centroids']))]
    for cluster_ix in range(1, len(m['unique_centroids']) + 1):
        c_nix = rowix(m, m['unique_centroids'][cluster_ix - 1])
        for i in range(1, m['data'].shape[0] + 1):
            coord_obj = coordinate(m, m['fitted_obs'][i - 1])
            c_obj = m['centroids'][coord_obj.x - 1][coord_obj.y - 1]
            c_obj_nix = rowix(m, c_obj)
            if c_obj_nix == c_nix:
                centroid_obs[cluster_ix - 1].append(i)
    return centroid_obs

def compute_wcss(m):
    clusters_ss = []
    for cluster_ix in range(1, len(m['unique_centroids']) + 1):
        c_nix = rowix(m, m['unique_centroids'][cluster_ix - 1])
        vectors = m['neurons'].iloc[[c_nix - 1]].values
        for obs in m['centroid_obs'][cluster_ix - 1]:
            vectors = np.vstack([vectors, m['data'].iloc[obs - 1].values])
        d_matrix = squareform(pdist(vectors))
        distances = d_matrix[0, :]
        distances_sqd = np.square(distances)
        c_ss = np.sum(distances_sqd) / (len(distances_sqd) - 1) if len(distances_sqd) > 1 else 0
        clusters_ss.append(c_ss)
    wcss = np.sum(clusters_ss) / len(clusters_ss) if clusters_ss else 0
    return wcss

def compute_bcss(m):
    all_bc_ss = []
    first_centroid_nix = rowix(m, m['unique_centroids'][0])
    cluster_vectors = m['neurons'].iloc[[first_centroid_nix - 1]].values
    if len(m['unique_centroids']) > 1:
        for cluster_ix in range(2, len(m['unique_centroids']) + 1):
            c_nix = rowix(m, m['unique_centroids'][cluster_ix - 1])
            c_vector = m['neurons'].iloc[[c_nix - 1]].values
            cluster_vectors = np.vstack([cluster_vectors, c_vector])
    for cluster_ix in range(1, len(m['unique_centroids']) + 1):
        c_nix = rowix(m, m['unique_centroids'][cluster_ix - 1])
        c_vector = m['neurons'].iloc[[c_nix - 1]].values
        compute_vectors = np.vstack([c_vector, cluster_vectors])
        d_matrix = squareform(pdist(compute_vectors))
        bc_distances = d_matrix[0, :]
        bc_distances_sqd = np.square(bc_distances)
        bc_ss = np.sum(bc_distances_sqd) / (len(bc_distances_sqd) - 2) if len(bc_distances_sqd) > 2 else 0
        all_bc_ss.append(bc_ss)
    assert len(all_bc_ss) == len(m['unique_centroids'])
    bcss = np.sum(all_bc_ss) / len(all_bc_ss) if all_bc_ss else 0
    return bcss

def compute_label_to_centroid(m):
    conv = {}
    for i, centroid in enumerate(m['unique_centroids'], start=1):
        x = centroid.x
        y = centroid.y
        l = m['centroid_labels'][x - 1][y - 1]
        if l not in conv:
            conv[l] = [i]
        else:
            conv[l].append(i)
    return conv

def map_fitted_obs(m):
    fitted_obs = []
    for i in range(1, m['data'].shape[0] + 1):
        b = best_match(m, m['data'].iloc[i - 1].values)
        fitted_obs.append(b)
    return fitted_obs

def map_topo(m, k=50, conf_int=0.95, verb=False, interval=True):
    data_arr = m['data'].values
    k = min(k, data_arr.shape[0])
    data_sample_ix = np.random.choice(range(data_arr.shape[0]), size=k, replace=False)
    acc_v = []
    for i in range(k):
        acc_v.append(accuracy(m, data_arr[data_sample_ix[i]], data_sample_ix[i] + 1))
    if interval:
        bval = bootstrap(m, conf_int, data_arr, k, acc_v)
    if verb:
        return acc_v
    else:
        val = np.sum(acc_v) / k
        if interval:
            return {'val': val, 'lo': bval['lo'], 'hi': bval['hi']}
        else:
            return val

def map_embed_vm(m, conf_int=0.95, verb=False):
    map_df = m['neurons']
    data_df = m['data']
    nfeatures = map_df.shape[1]
    vl = df_var_test(map_df, data_df, conf=conf_int)
    ml = df_mean_test(map_df, data_df, conf=conf_int)
    prob_v = map_significance(m, graphics=False)
    var_sum = 0
    for i in range(nfeatures):
        if (vl['conf_int_lo'][i] <= 1.0 and vl['conf_int_hi'][i] >= 1.0 and
            ml['conf_int_lo'][i] <= 0.0 and ml['conf_int_hi'][i] >= 0.0):
            var_sum += prob_v[i]
        else:
            prob_v[i] = 0
    if verb:
        return prob_v
    else:
        return var_sum

def map_embed_ks(m, conf_int=0.95, verb=False):
    map_df = m['neurons']
    data_df = m['data']
    nfeatures = map_df.shape[1]
    ks_vector = []
    for i in range(nfeatures):
        result = ks_2samp(map_df.iloc[:, i], data_df.iloc[:, i])
        ks_vector.append(result)
    prob_v = map_significance(m, graphics=False)
    var_sum = 0
    for i in range(nfeatures):
        if ks_vector[i].pvalue > (1 - conf_int):
            var_sum += prob_v.iloc[i]
        else:
            prob_v.iloc[i] = 0
    if verb:
        return prob_v
    else:
        return var_sum

def map_normalize(x):
    if isinstance(x, (list, np.ndarray, pd.Series)):
        arr = np.array(x, dtype=float)
        return (arr - arr.mean()) / arr.std() if arr.std() != 0 else arr - arr.mean()
    elif isinstance(x, pd.DataFrame):
        return x.apply(lambda row: (row - row.mean()) / row.std() if row.std() != 0 else row - row.mean(), axis=1)
    else:
        raise ValueError("'x' is not a vector or dataframe.")

def bootstrap(m, conf_int, data_arr, k, sample_acc_v):
    ix = int(100 - conf_int * 100)
    bn = 200
    bootstrap_acc_v = [np.sum(sample_acc_v) / k]
    for i in range(1, bn):
        bs_indices = np.random.choice(range(k), size=k, replace=True)
        a = np.sum(np.array(sample_acc_v)[bs_indices]) / k
        bootstrap_acc_v.append(a)
    bootstrap_acc_sort_v = np.sort(bootstrap_acc_v)
    lo_val = bootstrap_acc_sort_v[ix]
    hi_val = bootstrap_acc_sort_v[bn - ix - 1]
    return {'lo': lo_val, 'hi': hi_val}

def best_match(m, obs, full=False):
    obs = np.array(obs, dtype=float)
    neurons = m['neurons'].values.astype(float)
    diff = neurons - obs
    squ = diff ** 2
    s = np.sum(squ, axis=1)
    d = np.sqrt(s)
    order = np.argsort(d) + 1  # 1-indexing
    if full:
        return order.tolist()
    else:
        return int(order[0])

def accuracy(m, sample, data_ix):
    order = best_match(m, sample, full=True)
    best_ix = order[0]
    second_best_ix = order[1]
    coord_best = coordinate(m, best_ix)
    coord_x = coord_best.x
    coord_y = coord_best.y
    map_ix = m['fitted_obs'][data_ix - 1]
    coord_map = coordinate(m, map_ix)
    map_x = coord_map.x
    map_y = coord_map.y
    if coord_x != map_x or coord_y != map_y or best_ix != map_ix:
        print("best_ix: ", best_ix, " map.rix: ", map_ix)
        raise ValueError("problems with coordinates")
    best_xy = coordinate(m, best_ix)
    second_best_xy = coordinate(m, second_best_ix)
    diff_map = np.array([best_xy.x, best_xy.y]) - np.array([second_best_xy.x, second_best_xy.y])
    dist_map = np.sqrt(np.sum(diff_map ** 2))
    return 1 if dist_map < 2 else 0

def coordinate(m, rowix):
    # Convert 1-indexed rowix to (x,y) coordinates on the map
    x = ((rowix - 1) % m['xdim']) + 1
    y = ((rowix - 1) // m['xdim']) + 1
    return Coord(x, y)

def rowix(m, cd):
    if not isinstance(cd, Coord):
        raise ValueError("expected a Coord object")
    return cd.x + (cd.y - 1) * m['xdim']

def map_graphics_set():
    # Save current matplotlib rcParams
    return plt.rcParams.copy()

def map_graphics_reset(params):
    plt.rcParams.update(params)

def plot_heat(m):
    x = m['xdim']
    y = m['ydim']
    centroids = m['centroids']
    if x <= 1 or y <= 1:
        raise ValueError("map dimensions too small")
    heat_arr = m['heat'].flatten()
    bins = np.linspace(heat_arr.min(), heat_arr.max(), 101)
    heat_v = np.digitize(heat_arr, bins)
    heat_matrix = heat_v.reshape(x, y)
    #cmap = plt.get_cmap("hot", 100)
    cmap = plt.get_cmap("plasma", 100)

    orig_params = map_graphics_set()
    plt.figure()
    plt.xlim(0, x)
    plt.ylim(0, y)
    plt.box(False)
    plt.xlabel("x")
    plt.ylabel("y")
    xticks = [0.5, x-0.5] 
    yticks = [0.5, y-0.5] 
    xlabels = [1, x] 
    ylabels = [1, y] 
    #xticks = np.arange(0.5, x, 1)
    #yticks = np.arange(0.5, y, 1)
    #xlabels = np.arange(1, x + 1,1)
    #ylabels = np.arange(1, y + 1,1)
    plt.xticks(xticks, xlabels)
    plt.yticks(yticks, ylabels)
    for ix in range(1, x + 1):
        for iy in range(1, y + 1):
            color_index = 100 - heat_matrix[ix - 1, iy - 1]
            rect = plt.Rectangle((ix-1, iy-1), 1, 1, color=cmap(color_index / 100), ec=None)
            plt.gca().add_patch(rect)
    for ix in range(1, x + 1):
        for iy in range(1, y + 1):
            c = centroids[ix - 1][iy - 1]
            plt.plot([ix - 0.5, c.x - 0.5], 
                     [iy - 0.5, c.y - 0.5], 
                     linewidth=0.5,
                     color="grey")
    centroid_labels = majority_labels(m)
    for ix in range(1, x + 1):
        for iy in range(1, y + 1):
            lab = centroid_labels[ix - 1][iy - 1]
            if lab != "<None>":
                plt.text(ix - 0.5, 
                         iy - 0.5, 
                         str(lab), 
                         ha="center", 
                         va="center",
                         size="x-small",
                         color="black")
    plt.show()
    map_graphics_reset(orig_params)

def compute_centroids(m):
    heat = m['heat']
    xdim = m['xdim']
    ydim = m['ydim']
    centroids = [[Coord() for _ in range(ydim)] for _ in range(xdim)]
    def compute_centroid(ix, iy):
        if centroids[ix - 1][iy - 1].x > -1 and centroids[ix - 1][iy - 1].y > -1:
            return centroids[ix - 1][iy - 1]
        min_val = heat[ix - 1, iy - 1]
        min_x, min_y = ix, iy
        # For inner nodes
        if ix > 1 and ix < xdim and iy > 1 and iy < ydim:
            for dx, dy in [(-1, -1), (0, -1), (1, -1),
                           (1, 0), (1, 1), (0, 1),
                           (-1, 1), (-1, 0)]:
                nx, ny = ix + dx, iy + dy
                if heat[nx - 1, ny - 1] < min_val:
                    min_val = heat[nx - 1, ny - 1]
                    min_x, min_y = nx, ny
        else:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx = ix + dx
                    ny = iy + dy
                    if 1 <= nx <= xdim and 1 <= ny <= ydim:
                        if heat[nx - 1, ny - 1] < min_val:
                            min_val = heat[nx - 1, ny - 1]
                            min_x, min_y = nx, ny
        if min_x != ix or min_y != iy:
            return compute_centroid(min_x, min_y)
        else:
            return Coord(ix, iy)
    for i in range(1, xdim + 1):
        for j in range(1, ydim + 1):
            centroids[i - 1][j - 1] = compute_centroid(i, j)
    return centroids

def compute_heat(m):
    neurons = m['neurons'].values
    d_matrix = squareform(pdist(neurons))
    x = m['xdim']
    y = m['ydim']
    if x == 1 or y == 1:
        raise ValueError("heat map cannot be computed for a map of dimension 1")
    heat = np.zeros((x, y))
    def xl(ix, iy):
        return rowix(m, Coord(ix, iy))
    if x > 2 and y > 2:
        for ix in range(2, x):
            for iy in range(2, y):
                s = (d_matrix[xl(ix, iy) - 1, xl(ix - 1, iy - 1) - 1] +
                     d_matrix[xl(ix, iy) - 1, xl(ix, iy - 1) - 1] +
                     d_matrix[xl(ix, iy) - 1, xl(ix + 1, iy - 1) - 1] +
                     d_matrix[xl(ix, iy) - 1, xl(ix + 1, iy) - 1] +
                     d_matrix[xl(ix, iy) - 1, xl(ix + 1, iy + 1) - 1] +
                     d_matrix[xl(ix, iy) - 1, xl(ix, iy + 1) - 1] +
                     d_matrix[xl(ix, iy) - 1, xl(ix - 1, iy + 1) - 1] +
                     d_matrix[xl(ix, iy) - 1, xl(ix - 1, iy) - 1])
                heat[ix - 1, iy - 1] = s / 8
        for ix in range(2, x):
            iy = 1
            s = (d_matrix[xl(ix, iy) - 1, xl(ix + 1, iy) - 1] +
                 d_matrix[xl(ix, iy) - 1, xl(ix + 1, iy + 1) - 1] +
                 d_matrix[xl(ix, iy) - 1, xl(ix, iy + 1) - 1] +
                 d_matrix[xl(ix, iy) - 1, xl(ix - 1, iy + 1) - 1] +
                 d_matrix[xl(ix, iy) - 1, xl(ix - 1, iy) - 1])
            heat[ix - 1, iy - 1] = s / 5
        for ix in range(2, x):
            iy = y
            s = (d_matrix[xl(ix, iy) - 1, xl(ix - 1, iy - 1) - 1] +
                 d_matrix[xl(ix, iy) - 1, xl(ix, iy - 1) - 1] +
                 d_matrix[xl(ix, iy) - 1, xl(ix + 1, iy - 1) - 1] +
                 d_matrix[xl(ix, iy) - 1, xl(ix + 1, iy) - 1] +
                 d_matrix[xl(ix, iy) - 1, xl(ix - 1, iy) - 1])
            heat[ix - 1, iy - 1] = s / 5
        for iy in range(2, y):
            ix = 1
            s = (d_matrix[xl(ix, iy) - 1, xl(ix, iy - 1) - 1] +
                 d_matrix[xl(ix, iy) - 1, xl(ix + 1, iy - 1) - 1] +
                 d_matrix[xl(ix, iy) - 1, xl(ix + 1, iy) - 1] +
                 d_matrix[xl(ix, iy) - 1, xl(ix + 1, iy + 1) - 1] +
                 d_matrix[xl(ix, iy) - 1, xl(ix, iy + 1) - 1])
            heat[ix - 1, iy - 1] = s / 5
        for iy in range(2, y):
            ix = x
            s = (d_matrix[xl(ix, iy) - 1, xl(ix - 1, iy - 1) - 1] +
                 d_matrix[xl(ix, iy) - 1, xl(ix, iy - 1) - 1] +
                 d_matrix[xl(ix, iy) - 1, xl(ix, iy + 1) - 1] +
                 d_matrix[xl(ix, iy) - 1, xl(ix - 1, iy + 1) - 1] +
                 d_matrix[xl(ix, iy) - 1, xl(ix - 1, iy) - 1])
            heat[ix - 1, iy - 1] = s / 5
    if x >= 2 and y >= 2:
        ix, iy = 1, 1
        s = (d_matrix[xl(ix, iy) - 1, xl(ix + 1, iy) - 1] +
             d_matrix[xl(ix, iy) - 1, xl(ix + 1, iy + 1) - 1] +
             d_matrix[xl(ix, iy) - 1, xl(ix, iy + 1) - 1])
        heat[ix - 1, iy - 1] = s / 3

        ix, iy = x, 1
        s = (d_matrix[xl(ix, iy) - 1, xl(ix, iy + 1) - 1] +
             d_matrix[xl(ix, iy) - 1, xl(ix - 1, iy + 1) - 1] +
             d_matrix[xl(ix, iy) - 1, xl(ix - 1, iy) - 1])
        heat[ix - 1, iy - 1] = s / 3

        ix, iy = 1, y
        s = (d_matrix[xl(ix, iy) - 1, xl(ix, iy - 1) - 1] +
             d_matrix[xl(ix, iy) - 1, xl(ix + 1, iy - 1) - 1] +
             d_matrix[xl(ix, iy) - 1, xl(ix + 1, iy) - 1])
        heat[ix - 1, iy - 1] = s / 3

        ix, iy = x, y
        s = (d_matrix[xl(ix, iy) - 1, xl(ix - 1, iy - 1) - 1] +
             d_matrix[xl(ix, iy) - 1, xl(ix, iy - 1) - 1] +
             d_matrix[xl(ix, iy) - 1, xl(ix - 1, iy) - 1])
        heat[ix - 1, iy - 1] = s / 3
    heat_smoothed = gaussian_filter(heat, sigma=2)
    return heat_smoothed

def df_var_test(df1, df2, conf=0.95):
    if df1.shape[1] != df2.shape[1]:
        raise ValueError("cannot compare variances of data frames")
    n = df1.shape[1]
    var_ratio_v = np.ones(n)
    conf_int_lo_v = np.ones(n)
    conf_int_hi_v = np.ones(n)
    for i in range(n):
        s1 = df1.iloc[:, i].var()
        s2 = df2.iloc[:, i].var()
        ratio = s1 / s2 if s2 != 0 else np.inf
        var_ratio_v[i] = ratio
        dfn = df1.shape[0] - 1
        dfd = df2.shape[0] - 1
        alpha = 1 - conf
        lower = f.ppf(alpha / 2, dfn, dfd)
        upper = f.ppf(1 - alpha / 2, dfn, dfd)
        conf_int_lo_v[i] = ratio / upper
        conf_int_hi_v[i] = ratio / lower if lower != 0 else np.inf
    return {'ratio': var_ratio_v, 'conf_int_lo': conf_int_lo_v, 'conf_int_hi': conf_int_hi_v}

def df_mean_test(df1, df2, conf=0.95):
    if df1.shape[1] != df2.shape[1]:
        raise ValueError("cannot compare means of data frames")
    n = df1.shape[1]
    mean_diff_v = np.ones(n)
    conf_int_lo_v = np.ones(n)
    conf_int_hi_v = np.ones(n)
    for i in range(n):
        res = ttest_ind(df1.iloc[:, i], df2.iloc[:, i], equal_var=False)
        mean_diff_v[i] = df1.iloc[:, i].mean() - df2.iloc[:, i].mean()
        conf_int_lo_v[i] = -np.inf
        conf_int_hi_v[i] = np.inf
    return {'diff': mean_diff_v, 'conf_int_lo': conf_int_lo_v, 'conf_int_hi': conf_int_hi_v}

def get_unique_centroids(m):
    centroids = m['centroids']
    xdim = m['xdim']
    ydim = m['ydim']
    cd_list = []
    for ix in range(1, xdim + 1):
        for iy in range(1, ydim + 1):
            c_xy = centroids[ix - 1][iy - 1]
            if not any(x.x == c_xy.x and x.y == c_xy.y for x in cd_list):
                cd_list.append(c_xy)
    return cd_list

none_label = "<None>"

def majority_labels(m):
    if m['labels'] is None:
        return numerical_labels(m)
    x = m['xdim']
    y = m['ydim']
    nobs = m['data'].shape[0]
    centroid_labels_list = [[[] for _ in range(y)] for _ in range(x)]
    majority_labels_array = [[none_label for _ in range(y)] for _ in range(x)]
    for i in range(1, nobs + 1):
        lab = str(m['labels'].iloc[i - 1, 0])
        nix = m['fitted_obs'][i - 1]
        c = coordinate(m, nix)
        ix, iy = c.x, c.y
        centroid = m['centroids'][ix - 1][iy - 1]
        cx, cy = centroid.x, centroid.y
        centroid_labels_list[cx - 1][cy - 1].append(lab)
    for ix in range(1, x + 1):
        for iy in range(1, y + 1):
            label_v = centroid_labels_list[ix - 1][iy - 1]
            if len(label_v) != 0:
                counts = pd.Series(label_v).value_counts()
                majority_labels_array[ix - 1][iy - 1] = counts.index[0]
    return majority_labels_array

def numerical_labels(m):
    label_cnt = 1
    centroids = m['centroids']
    unique_centroids = m['unique_centroids']
    x = m['xdim']
    y = m['ydim']
    centroid_labels_array = [[none_label for _ in range(y)] for _ in range(x)]
    for uc in unique_centroids:
        label = f"centroid {label_cnt}"
        label_cnt += 1
        ix, iy = uc.x, uc.y
        centroid_labels_array[ix - 1][iy - 1] = label
    return centroid_labels_array

#---------------------------
# Research functions
#---------------------------
def compute_nwcss(m):
    clusters_ss = []
    for cluster_ix in range(1, len(m['unique_centroids']) + 1):
        c_nix = rowix(m, m['unique_centroids'][cluster_ix - 1])
        vectors = m['neurons'].iloc[[c_nix - 1]].values
        for obs in m['centroid_obs'][cluster_ix - 1]:
            obs_nix = m['fitted_obs'][obs - 1]
            obs_coord = coordinate(m, obs_nix)
            centroid_coord = m['centroids'][obs_coord.x - 1][obs_coord.y - 1]
            centroid_nix = rowix(m, centroid_coord)
            if centroid_nix == c_nix:
                vectors = np.vstack([vectors, m['neurons'].iloc[obs_nix - 1].values])
        d_matrix = squareform(pdist(vectors))
        distances = d_matrix[0, :]
        distances_sqd = np.square(distances)
        c_ss = np.sum(distances_sqd) / (len(distances_sqd) - 1) if len(distances_sqd) > 1 else 0
        clusters_ss.append(c_ss)
    wcss = np.sum(clusters_ss) / len(clusters_ss) if clusters_ss else 0
    return wcss

def avg_homogeneity(m):
    if m['labels'] is None:
        raise ValueError("you need to attach labels to the map")
    if m['xdim'] <= 1 or m['ydim'] <= 1:
        raise ValueError("map dimensions too small")
    x = m['xdim']
    y = m['ydim']
    centroids = m['centroids']
    nobs = m['data'].shape[0]
    centroid_labels_list = [[[] for _ in range(y)] for _ in range(x)]
    for i in range(1, nobs + 1):
        lab = str(m['labels'].iloc[i - 1, 0])
        nix = m['fitted_obs'][i - 1]
        c = coordinate(m, nix)
        ix, iy = c.x, c.y
        centroid = centroids[ix - 1][iy - 1]
        cx, cy = centroid.x, centroid.y
        centroid_labels_list[cx - 1][cy - 1].append(lab)
    sum_majority = 0
    n_centroids = 0
    for ix in range(1, x + 1):
        for iy in range(1, y + 1):
            label_v = centroid_labels_list[ix - 1][iy - 1]
            if len(label_v) != 0:
                n_centroids += 1
                counts = pd.Series(label_v).value_counts()
                m_val = counts.iloc[0]
                sum_majority += m_val
    return {'homog': sum_majority / nobs, 'nclust': n_centroids}

#---------------------------
# Vsom implementation
#---------------------------

def coord2D(ix, xdim):
    """
    Convert a 1D neuron index (0-indexed) into a 2D map coordinate.
    
    Parameters:
        ix (int): The 0-indexed neuron index.
        xdim (int): The width of the grid.
        
    Returns:
        tuple: (x, y) coordinate on the grid.
    """
    # For Fortran, the conversion was:
    #   coord(1) = modulo(ix-1, xdim) + 1
    #   coord(2) = (ix-1)//xdim + 1
    # In Python we use 0-indexing:
    return (ix % xdim, ix // xdim)

def Gamma(cache, cache_valid, coord_lookup, nsize, xdim, ydim, c):
    """
    Compute (and cache) the neighborhood vector for neuron 'c'.
    
    The neighborhood vector is 1.0 for neurons whose squared Euclidean distance 
    (on the grid) from neuron 'c' is below a threshold ((nsize * 1.5)**2),
    and 0.0 otherwise.
    
    Parameters:
        cache (ndarray): A 2D array of shape (num_neurons, num_neurons) that holds 
                         neighborhood vectors in its columns.
        cache_valid (ndarray): A boolean array of length num_neurons that tracks 
                               whether the neighborhood for a given neuron is up-to-date.
        coord_lookup (ndarray): An array of shape (num_neurons, 2) with each neuron’s grid coordinates.
        nsize (int): The current neighborhood size (affecting the threshold).
        xdim (int): The width of the grid.
        ydim (int): The height of the grid.
        c (int): The index of the winning neuron.
    """
    # If the neighborhood vector for neuron c has been computed already, do nothing.
    if cache_valid[c]:
        return

    # Get the (x,y) coordinate for neuron c.
    c2D = coord_lookup[c]  # this is a 1D array: [x, y]

    # Compute the squared Euclidean distances from c2D to every neuron’s coordinate.
    d = np.sum((coord_lookup - c2D)**2, axis=1)
    
    # Determine the threshold. (nsize*1.5)**2 in Fortran.
    threshold = (nsize * 1.5) ** 2
    
    # Set the neighborhood vector: 1.0 where distance is below threshold, else 0.0.
    cache[:, c] = np.where(d < threshold, 1.0, 0.0)
    
    # Mark this column as valid.
    cache_valid[c] = True

def vsom(dt, xdim, ydim, alpha, train):
    """
    Implements a stochastic Self-Organizing Map (SOM) training algorithm.
    
    Parameters:
        dt (ndarray or DataFrame): The training data as an array of shape (dtrows, dtcols) or as a pandas DataFrame.
        dtix (array-like): A sequence (length 'train') of indices into dt representing the order 
                           in which training observations are selected.
                           (Indices are assumed to be 0-indexed.)
        xdim (int): The width (number of columns) of the neuron grid.
        ydim (int): The height (number of rows) of the neuron grid.
        alpha (float): The learning rate.
        train (int): The number of training iterations (epochs).
    
    Returns:
        ndarray: The neuron weights after training.
    """
    # Initialize neurons with small random values.
    neurons = np.random.rand(xdim*ydim, dt.shape[1]).astype(np.float32) * 0.1
    num_neurons = neurons.shape[0]

    # Initialize data selector
    dtix = np.random.choice(dt.shape[0], train, replace=True)

    # If dt is a DataFrame, convert it to a NumPy array.
    if isinstance(dt, pd.DataFrame):
        dt = dt.values
    
    # setup: determine initial neighborhood size and steps.
    nsize = max(xdim, ydim) + 1
    nsize_step = int(np.ceil(train / nsize))
    step_counter = 0

    # Initialize the neighborhood cache:
    # 'cache' is a 2D array of shape (num_neurons, num_neurons) where each column will hold a neighborhood vector.
    cache = np.zeros((num_neurons, num_neurons), dtype=np.float32)
    # 'cache_valid' keeps track of which neuron's neighborhood vector is valid.
    cache_valid = np.zeros(num_neurons, dtype=bool)

    # Build the 2D coordinate lookup table.
    coord_lookup = np.zeros((num_neurons, 2), dtype=np.int32)
    for i in range(num_neurons):
        coord_lookup[i, :] = np.array(coord2D(i, xdim))
    
    # Training loop over epochs.
    for epoch in range(train):
        step_counter += 1
        if step_counter == nsize_step:
            step_counter = 0
            nsize -= 1
            # Invalidate all cached neighborhood vectors.
            cache_valid[:] = False

        # Select a training observation using the provided index order.
        ix = dtix[epoch]  # dtix should be 0-indexed.
        
        # Competitive step:
        # Compute the difference between every neuron's weight vector and the chosen training sample.
        diff = neurons - dt[ix, :]  # Broadcasting dt[ix, :] over all neurons.
        # Compute squared Euclidean distances.
        s = np.sum(diff**2, axis=1)
        # Determine the index 'c' of the neuron with the smallest distance.
        c = np.argmin(s)
        
        # Update step:
        # Compute (and cache) the neighborhood vector for the winning neuron.
        Gamma(cache, cache_valid, coord_lookup, nsize, xdim, ydim, c)
        # Compute a time-varying factor beta (decreases with epochs).
        beta = 1.0 - (epoch + 1) / train  # Note: Fortran epochs start at 1.
        # Update each neuron's weight vector.
        # The update is: neuron = neuron - (neighborhood_value * alpha * beta) * diff.
        neurons = neurons - (cache[:, c][:, np.newaxis] * alpha * beta) * diff

    return neurons

# test code
if __name__ == "__main__":
    from sklearn import datasets
    import time

    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.DataFrame(iris.target_names[iris.target],columns=['species'])

    # Build the map
    start = time.time()
    som_map = map_build(X, labels=y, xdim=20, ydim=15, train=1000000, seed=42)
    end = time.time()
    print(f"Time elapsed: {end - start} seconds")

    map_summary(som_map)
    map_starburst(som_map)
    print(map_significance(som_map))
    map_marginal(som_map,2)
    v = map_summary(som_map, verb=False)
    print(v['quality_assessments']['convergence'].iloc[0])
    print(som_map['unique_centroids'])

