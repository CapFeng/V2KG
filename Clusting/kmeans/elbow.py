import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

def main():
    cluster1 = np.random.uniform(0.5, 1.5, (2, 10))
    cluster2 = np.random.uniform(5.5, 6.5, (2, 10))
    cluster3 = np.random.uniform(3.0, 4.0, (2, 10))
    X = np.hstack((cluster1, cluster2, cluster3)).T
    plt.scatter(X[:, 0], X[:, 1])
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

    K = range(1, 10)
    meandistortions = []
    for k in K:
        kmean = KMeans(n_clusters=k)
        kmean.fit(X)
        meandistortions.append(sum(np.min(cdist(X, kmean.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
    plt.plot(K, meandistortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Averange Dispersion')
    plt.title('Selecting k with the Elbow Method')
    plt.show()

if __name__ == "__main__":
    main()