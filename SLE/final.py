import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class PreProcessing:
    @staticmethod
    def data_description(data):
        print(f'Data Shape: {data.shape}')
        print('Data Info:')
        data.info()
        print(f'Data Description:\n{data.describe()}')
        print(f'Null Values:\n{data.isnull().sum()}')

    @staticmethod
    def drop_na_columns(data):
        return data.dropna(axis=1, how='all')

    @staticmethod
    def impute_missing_values(data):
        return SimpleImputer(strategy='mean').fit_transform(data)

    @staticmethod
    def normalize(x):
        return StandardScaler().fit_transform(x)

    @staticmethod
    def apply_pca(x, n_components):
        pca = PCA(n_components=n_components)
        x_pca = pca.fit_transform(x)
        print(f"Explained variance: {pca.explained_variance_ratio_}, Total: {sum(pca.explained_variance_ratio_)}")
        return x_pca

    @staticmethod
    def apply_lda(x, y, n_components=None):
        lda = LinearDiscriminantAnalysis(n_components=n_components)
        return lda.fit_transform(x, y)


class EvaluationMetrics:
    @staticmethod
    def plot_clusters(x, labels, title):
        plt.scatter(x[:, 0], x[:, 1], c=labels, cmap='viridis', edgecolor='k', s=50)
        plt.title(title)
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.colorbar(label='Cluster Label')
        plt.show()

    @staticmethod
    def silhouette_score(x, labels):
        return silhouette_score(x, labels) if len(set(labels)) > 1 else -1

    @staticmethod
    def calinski_harabasz_score(x, labels):
        return calinski_harabasz_score(x, labels) if len(set(labels)) > 1 else -1

    @staticmethod
    def davies_bouldin_score(x, labels):
        return davies_bouldin_score(x, labels) if len(set(labels)) > 1 else -1

    @staticmethod
    def elbow_method(x, max_clusters=10):
        distortions = [KMeans(n_clusters=i).fit(x).inertia_ for i in range(1, max_clusters + 1)]
        plt.plot(range(1, max_clusters + 1), distortions, marker='o')
        plt.title('Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('Distortion')
        plt.grid()
        plt.show()


class Clustering:
    @staticmethod
    def kMeans(x, n_clusters):
        labels = KMeans(n_clusters=n_clusters).fit_predict(x)
        EvaluationMetrics.plot_clusters(x, labels, 'KMeans')
        return labels

    @staticmethod
    def agglomerative_clustering(x, n_clusters):
        labels = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(x)
        EvaluationMetrics.plot_clusters(x, labels, 'Agglomerative')
        return labels

    @staticmethod
    def spectral_clustering(x, n_clusters):
        labels = SpectralClustering(n_clusters=n_clusters).fit_predict(x)
        EvaluationMetrics.plot_clusters(x, labels, 'Spectral')
        return labels

    @staticmethod
    def dbScan(x):
        labels = DBSCAN(eps=0.5, min_samples=5).fit_predict(x)
        EvaluationMetrics.plot_clusters(x, labels, 'DBSCAN')
        return labels

    @staticmethod
    def clustering_report(x, n_clusters):
        return {
            'KMeans': Clustering.kMeans(x, n_clusters),
            'Agglomerative': Clustering.agglomerative_clustering(x, n_clusters),
            'Spectral': Clustering.spectral_clustering(x, n_clusters),
            'DBSCAN': Clustering.dbScan(x)
        }


def display_scores(scores, method_name, title):
    print(f"\n### {title} ({method_name}):")
    print(f"- Silhouette Score: {scores['Silhouette Score']:.4f}")
    print(f"- Calinski-Harabasz Score: {scores['Calinski-Harabasz Score']:.2f}")
    print(f"- Davies-Bouldin Score: {scores['Davies-Bouldin Score']:.2f}\n")


def evaluate_clusters(x, cluster_labels, title="PCA with 2 components"):
    metrics = ['Silhouette Score', 'Calinski-Harabasz Score', 'Davies-Bouldin Score']
    scores = {method: {} for method in cluster_labels}

    for method, labels in cluster_labels.items():
        scores[method]['Silhouette Score'] = EvaluationMetrics.silhouette_score(x, labels)
        scores[method]['Calinski-Harabasz Score'] = EvaluationMetrics.calinski_harabasz_score(x, labels)
        scores[method]['Davies-Bouldin Score'] = EvaluationMetrics.davies_bouldin_score(x, labels)

        display_scores(scores[method], method, title)

    return scores


# Load and preprocess data
data = pd.read_csv("./dataset/plant.csv")
data = data.drop('CUST_ID', axis=1)
PreProcessing.data_description(data)
x = PreProcessing.drop_na_columns(data)
x = pd.get_dummies()
x = PreProcessing.normalize(PreProcessing.impute_missing_values(x))

# Apply PCA for 2 to 4 components and perform clustering
max_k = 5
for n_components in range(2, 4):
    print(f"\nApplying PCA with {n_components} components.")
    x_pca = PreProcessing.apply_pca(x, n_components)
    
    # Elbow method to find optimal k
    EvaluationMetrics.elbow_method(x_pca, max_k)
    optimal_k = int(input("Enter the optimal number of clusters: "))

    # Perform clustering and evaluate using PCA
    cluster_labels_pca = Clustering.clustering_report(x_pca, optimal_k)
    scores_pca = evaluate_clusters(x_pca, cluster_labels_pca, f"PCA with {n_components} components")

    # Perform clustering and evaluate using LDA
    print("\nApplying LDA for visualization:")
    for method, labels in cluster_labels_pca.items():
        x_lda = PreProcessing.apply_lda(x, labels, n_components=2)
        EvaluationMetrics.plot_clusters(x_lda, labels, f'{method} LDA')
        scores_lda = evaluate_clusters(x_lda, cluster_labels_pca, f"LDA with {n_components} components ({method} clusters)")
