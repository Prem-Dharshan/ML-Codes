import numpy as np
from pandas import read_csv
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

class PreProcessing:

    @staticmethod
    def data_description(data):
        print('Data Shape:', data.shape)
        print('Data Info:', data.info())
        print('Data Description:', data.describe())
        print('Data Head:', data.head())
        print('Data Tail:', data.tail())
        print('Data Columns:', data.columns)
        print('Data Null Values:', data.isnull().sum())
        sns.heatmap(data.isnull(), cbar=False)
        plt.show()
        sns.pairplot(data)
        plt.show()
        return None   

    @staticmethod
    def drop_na_columns(data):
        return data.dropna(axis=1, how='all')

    @staticmethod
    def impute_missing_values(data):
        imputer = SimpleImputer(strategy='mean')
        imputer.fit(data)
        return imputer.transform(data)

    @staticmethod
    def encode_categorical_x(x):
        te = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
        return te.fit_transform(x)
    
    @staticmethod
    def encode_categorical_y(y):
        le = LabelEncoder()
        return le.fit_transform(y)

    @staticmethod
    def normalize(x):
        scaler = StandardScaler()
        return scaler.fit_transform(x)

    @staticmethod
    def plot_labels_distribution(y):
        counter = Counter(y)
        labels = list(counter.keys())
        counts = list(counter.values())
        plt.figure(figsize=(8, 6))
        plt.bar(labels, counts, color=plt.cm.tab10.colors[:len(labels)], edgecolor='black')
        plt.xlabel('Labels')
        plt.ylabel('Count')
        plt.title('Label Distribution')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        return None

    @staticmethod
    def apply_pca(x, n_components=None):
        pca = PCA(n_components=n_components)
        x_pca = pca.fit_transform(x)
        explained_variance = pca.explained_variance_ratio_
        print(f"Explained variance by each component: {explained_variance}")
        print(f"Total explained variance: {sum(explained_variance)}")
        return x_pca, pca

    @staticmethod
    def plot_pca(x, y, title):
        plt.figure(figsize=(8, 6))
        plt.scatter(x[:, 0], x[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
        plt.title(f'PCA Plot - {title}')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.colorbar(label='Label')
        plt.show()
        return None


class EvaluationMetrics:

    @staticmethod
    def plot_clusters(x, labels, title):
        plt.figure(figsize=(8, 6))
        plt.scatter(x[:, 0], x[:, 1], c=labels, cmap='viridis', edgecolor='k', s=50)
        plt.title(f'Cluster Plot - {title}')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.colorbar(label='Cluster Label')
        plt.show()

    @staticmethod
    def silhouette_score(x, labels):
        if len(set(labels)) > 1:
            score = silhouette_score(x, labels)
        else:
            score = -1
        return score

    @staticmethod
    def calinski_harabasz_score(x, labels):
        if len(set(labels)) > 1:
            score = calinski_harabasz_score(x, labels)
        else:
            score = -1
        return score

    @staticmethod
    def davies_bouldin_score(x, labels):
        if len(set(labels)) > 1:
            score = davies_bouldin_score(x, labels)
        else:
            score = -1
        return score
    
    @staticmethod
    def elbow_method(x, max_clusters=10):
        distortions = []
        for i in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=i)
            kmeans.fit(x)
            distortions.append(kmeans.inertia_)
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, max_clusters + 1), distortions, marker='o')
        plt.title('Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('Distortion')
        plt.grid()
        plt.show()
        return distortions

class Clustering:

    @staticmethod
    def kMeans(x):
        kmeans = KMeans(n_clusters=2)
        kmeans.fit(x)
        cluster_labels_kmeans = kmeans.labels_
        print(f"Number of clusters formed: {len(np.unique(cluster_labels_kmeans))}")
        print(f"The labels assigned by KMeans are: {cluster_labels_kmeans}")
        EvaluationMetrics.plot_clusters(x, cluster_labels_kmeans, 'KMeans')
        return cluster_labels_kmeans

    @staticmethod
    def agglomerative_clustering(x):
        agg = AgglomerativeClustering(n_clusters=2)
        cluster_labels_agg = agg.fit_predict(x)
        print(f"Number of clusters formed: {len(np.unique(cluster_labels_agg))}")
        print(f"The labels assigned by Agglomerative Clustering are: {cluster_labels_agg}")
        EvaluationMetrics.plot_clusters(x, cluster_labels_agg, 'Agglomerative Clustering')
        return cluster_labels_agg

    @staticmethod
    def spectral_clustering(x):
        spectral = SpectralClustering(n_clusters=2)
        cluster_labels_spectral = spectral.fit_predict(x)
        print(f"Number of clusters formed: {len(np.unique(cluster_labels_spectral))}")
        print(f"The labels assigned by Spectral Clustering are: {cluster_labels_spectral}")
        EvaluationMetrics.plot_clusters(x, cluster_labels_spectral, 'Spectral Clustering')
        return cluster_labels_spectral

    @staticmethod
    def dbScan(x):
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        cluster_labels_dbscan = dbscan.fit_predict(x)
        print(f"Number of clusters formed: {len(np.unique(cluster_labels_dbscan))}")
        print(f"The labels assigned by DBSCAN are: {cluster_labels_dbscan}")
        EvaluationMetrics.plot_clusters(x, cluster_labels_dbscan, 'DBSCAN')
        return cluster_labels_dbscan

    @staticmethod
    def clustering_report(x):
        kmeans_labels = Clustering.kMeans(x)
        agg_labels = Clustering.agglomerative_clustering(x)
        spectral_labels = Clustering.spectral_clustering(x)
        dbscan_labels = Clustering.dbScan(x)
        return kmeans_labels, agg_labels, spectral_labels, dbscan_labels


def main() -> None:

    df = read_csv("./dataset/mall_customers.csv")
    df.drop('CustomerID', axis=1, inplace=True)

    PreProcessing.data_description(df)
    df = PreProcessing.drop_na_columns(df)

    df.ffill(inplace=True)
    
    x = df.iloc[:, :-1].values
    
    x = PreProcessing.encode_categorical_x(x)
    x = PreProcessing.impute_missing_values(x)
    x = PreProcessing.normalize(x)

    x_pca, pca = PreProcessing.apply_pca(x, n_components=2)

    kmeans_labels, agg_labels, spectral_labels, dbscan_labels = Clustering.clustering_report(x_pca)

    scores = {}

    scores['KMeans'] = EvaluationMetrics.silhouette_score(x_pca, kmeans_labels)
    scores['Agglomerative'] = EvaluationMetrics.silhouette_score(x_pca, agg_labels)
    scores['Spectral'] = EvaluationMetrics.silhouette_score(x_pca, spectral_labels)
    scores['DBSCAN'] = EvaluationMetrics.silhouette_score(x_pca, dbscan_labels)

    print(scores)

    best_model = max(scores, key=lambda x: scores[x])
    print(f"Best Model: {best_model}")
    
    return None



if '__main__' == __name__:
    main()
