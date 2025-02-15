import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from minisom import MiniSom
from scipy.spatial.distance import cdist
from scipy.stats import f_oneway, ttest_ind

class SOM:
    def __init__(self, data, xdim=10, ydim=5, alpha=0.3, train=1000, normalize=False, seed=None):
        self.data = data
        self.xdim = xdim
        self.ydim = ydim
        self.alpha = alpha
        self.train = train
        self.normalize = normalize
        self.seed = seed
        self.neurons = None
        self.labels = None
        self.heat = None
        self.fitted_obs = None
        self.centroids = None
        self.unique_centroids = None
        self.centroid_labels = None
        self.label_to_centroid = None
        self.centroid_obs = None
        self.convergence = None
        self.wcss = None
        self.bcss = None

        if self.normalize:
            self.data = self.normalize_data(self.data)

        self.build_map()

    def normalize_data(self, data):
        return (data - data.mean()) / data.std()

    def build_map(self):
        som = MiniSom(self.xdim, self.ydim, self.data.shape[1], sigma=1.0, learning_rate=self.alpha, random_seed=self.seed)
        som.train_random(self.data.values, self.train)
        self.neurons = som.get_weights().reshape(self.xdim * self.ydim, self.data.shape[1])
        self.neurons = pd.DataFrame(self.neurons, columns=self.data.columns)

        self.compute_heat()
        self.fitted_obs = self.map_fitted_obs()
        self.centroids = self.compute_centroids()
        self.unique_centroids = self.get_unique_centroids()
        self.centroid_labels = self.majority_labels()
        self.label_to_centroid = self.compute_label_to_centroid()
        self.centroid_obs = self.compute_centroid_obs()
        self.convergence = self.map_convergence()
        self.wcss = self.compute_wcss()
        self.bcss = self.compute_bcss()

    def compute_heat(self):
        distances = cdist(self.neurons, self.neurons)
        self.heat = distances.mean(axis=1).reshape(self.xdim, self.ydim)

    def map_fitted_obs(self):
        fitted_obs = []
        for i in range(len(self.data)):
            bmu = np.argmin(np.linalg.norm(self.neurons - self.data.iloc[i], axis=1))
            fitted_obs.append(bmu)
        return fitted_obs

    def compute_centroids(self):
        centroids = np.zeros((self.xdim, self.ydim, 2))
        for i in range(self.xdim):
            for j in range(self.ydim):
                centroids[i, j] = [i, j]
        return centroids

    def get_unique_centroids(self):
        unique_centroids = []
        for i in range(self.xdim):
            for j in range(self.ydim):
                if [i, j] not in unique_centroids:
                    unique_centroids.append([i, j])
        return unique_centroids

    def majority_labels(self):
        if self.labels is None:
            return self.numerical_labels()
        
        centroid_labels = np.empty((self.xdim, self.ydim), dtype=object)
        for i in range(len(self.data)):
            label = self.labels.iloc[i, 0]
            bmu = self.fitted_obs[i]
            x, y = self.coordinate(bmu)
            centroid_labels[x, y] = label
        return centroid_labels

    def numerical_labels(self):
        centroid_labels = np.empty((self.xdim, self.ydim), dtype=object)
        label_cnt = 1
        for i in range(self.xdim):
            for j in range(self.ydim):
                centroid_labels[i, j] = f"centroid {label_cnt}"
                label_cnt += 1
        return centroid_labels

    def compute_label_to_centroid(self):
        label_to_centroid = {}
        for i in range(len(self.unique_centroids)):
            x, y = self.unique_centroids[i]
            label = self.centroid_labels[x, y]
            if label not in label_to_centroid:
                label_to_centroid[label] = []
            label_to_centroid[label].append(i)
        return label_to_centroid

    def compute_centroid_obs(self):
        centroid_obs = [[] for _ in range(len(self.unique_centroids))]
        for i in range(len(self.data)):
            bmu = self.fitted_obs[i]
            x, y = self.coordinate(bmu)
            centroid_idx = self.unique_centroids.index([x, y])
            centroid_obs[centroid_idx].append(i)
        return centroid_obs

    def map_convergence(self, conf_int=0.95, k=50, verb=True, ks=True):
        if ks:
            embed = self.map_embed_ks(conf_int, verb=False)
        else:
            embed = self.map_embed_vm(conf_int, verb=False)

        topo = self.map_topo(k, conf_int, verb=False, interval=False)
        return 0.5 * embed + 0.5 * topo

    def map_embed_ks(self, conf_int=0.95, verb=False):
        from scipy.stats import ks_2samp
        prob_v = self.map_significance(graphics=False)
        var_sum = 0
        for i in range(self.data.shape[1]):
            ks_result = ks_2samp(self.data.iloc[:, i], self.neurons.iloc[:, i])
            if ks_result.pvalue > (1 - conf_int):
                var_sum += prob_v[i]
        return var_sum

    def map_embed_vm(self, conf_int=0.95, verb=False):
        prob_v = self.map_significance(graphics=False)
        var_sum = 0
        for i in range(self.data.shape[1]):
            f_test = f_oneway(self.data.iloc[:, i], self.neurons.iloc[:, i])
            t_test = ttest_ind(self.data.iloc[:, i], self.neurons.iloc[:, i])
            if f_test.pvalue > (1 - conf_int) and t_test.pvalue > (1 - conf_int):
                var_sum += prob_v[i]
        return var_sum

    def map_topo(self, k=50, conf_int=0.95, verb=False, interval=True):
        sample_indices = np.random.choice(len(self.data), size=k, replace=False)
        acc_v = []
        for i in sample_indices:
            acc_v.append(self.accuracy(self.data.iloc[i], i))
        if verb:
            return acc_v
        else:
            val = np.mean(acc_v)
            if interval:
                bval = self.bootstrap(conf_int, self.data.values, k, acc_v)
                return {'val': val, 'lo': bval['lo'], 'hi': bval['hi']}
            else:
                return val

    def accuracy(self, sample, data_idx):
        bmu = np.argmin(np.linalg.norm(self.neurons - sample, axis=1))
        second_bmu = np.argsort(np.linalg.norm(self.neurons - sample, axis=1))[1]
        dist_map = np.linalg.norm(np.array(self.coordinate(bmu)) - np.array(self.coordinate(second_bmu)))
        return 1 if dist_map < 2 else 0

    def bootstrap(self, conf_int, data, k, sample_acc_v):
        ix = int(100 - conf_int * 100)
        bn = 200
        bootstrap_acc_v = [np.mean(sample_acc_v)]
        for _ in range(1, bn):
            bs_v = np.random.choice(k, size=k, replace=True)
            a = np.mean(np.array(sample_acc_v)[bs_v])
            bootstrap_acc_v.append(a)
        bootstrap_acc_sort_v = np.sort(bootstrap_acc_v)
        lo_val = bootstrap_acc_sort_v[ix]
        hi_val = bootstrap_acc_sort_v[bn - ix]
        return {'lo': lo_val, 'hi': hi_val}

    def compute_wcss(self):
        clusters_ss = []
        for cluster_ix in range(len(self.unique_centroids)):
            centroid_idx = self.unique_centroids[cluster_ix]
            vectors = self.neurons.iloc[centroid_idx].values.reshape(1, -1)
            for obs_idx in self.centroid_obs[cluster_ix]:
                vectors = np.vstack([vectors, self.data.iloc[obs_idx].values])
            distances = np.linalg.norm(vectors - vectors[0], axis=1)
            distances_sqd = distances ** 2
            c_ss = np.sum(distances_sqd) / (len(distances_sqd) - 1)
            clusters_ss.append(c_ss)
        return np.mean(clusters_ss)

    def compute_bcss(self):
        all_bc_ss = []
        cluster_vectors = self.neurons.iloc[self.unique_centroids[0]].values.reshape(1, -1)
        for cluster_ix in range(1, len(self.unique_centroids)):
            centroid_idx = self.unique_centroids[cluster_ix]
            cluster_vectors = np.vstack([cluster_vectors, self.neurons.iloc[centroid_idx].values])
        for cluster_ix in range(len(self.unique_centroids)):
            centroid_idx = self.unique_centroids[cluster_ix]
            compute_vectors = np.vstack([self.neurons.iloc[centroid_idx].values, cluster_vectors])
            bc_distances = np.linalg.norm(compute_vectors - compute_vectors[0], axis=1)
            bc_distances_sqd = bc_distances ** 2
            bc_ss = np.sum(bc_distances_sqd) / (len(bc_distances_sqd) - 2)
            all_bc_ss.append(bc_ss)
        return np.mean(all_bc_ss)

    def coordinate(self, rowix):
        x = (rowix - 1) % self.xdim + 1
        y = (rowix - 1) // self.xdim + 1
        return x, y

    def map_significance(self, graphics=True, feature_labels=True):
        var_v = np.var(self.data, axis=0)
        var_sum = np.sum(var_v)
        prob_v = var_v / var_sum
        if graphics:
            plt.figure()
            plt.bar(range(len(prob_v)), prob_v)
            plt.xlabel("Features")
            plt.ylabel("Significance")
            if feature_labels:
                plt.xticks(range(len(prob_v)), self.data.columns, rotation=90)
            else:
                plt.xticks(range(len(prob_v)), range(1, len(prob_v) + 1))
            plt.show()
        return prob_v

    def map_marginal(self, marginal):
        if isinstance(marginal, int):
            train = pd.DataFrame({'points': self.data.iloc[:, marginal], 'legend': 'training data'})
            neurons = pd.DataFrame({'points': self.neurons.iloc[:, marginal], 'legend': 'neurons'})
            hist = pd.concat([train, neurons])
            sns.kdeplot(data=hist, x='points', hue='legend', fill=True, alpha=0.2)
            plt.xlabel(self.data.columns[marginal])
            plt.show()
        elif marginal in self.data.columns:
            train = pd.DataFrame({'points': self.data[marginal], 'legend': 'training data'})
            neurons = pd.DataFrame({'points': self.neurons[marginal], 'legend': 'neurons'})
            hist = pd.concat([train, neurons])
            sns.kdeplot(data=hist, x='points', hue='legend', fill=True, alpha=0.2)
            plt.xlabel(marginal)
            plt.show()
        else:
            raise ValueError("second argument is not a data dimension or index")

# Example usage:
# data = pd.DataFrame(np.random.rand(100, 5), columns=['A', 'B', 'C', 'D', 'E'])
# som = SOM(data)
# som.map_significance()
# som.map_marginal('A')