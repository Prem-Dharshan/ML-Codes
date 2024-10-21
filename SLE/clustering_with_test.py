import numpy as np
from pandas import read_csv, get_dummies, DataFrame
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import calinski_harabasz_score, confusion_matrix, davies_bouldin_score, make_scorer, precision_recall_curve, roc_auc_score, roc_curve, silhouette_score, accuracy_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from sklearn.metrics import classification_report
from ydata_profiling import ProfileReport
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
    def data_profiling(data):
        data = DataFrame(data) 
        profile = ProfileReport(data, title='Data Analysis Report', explorative=True)
        profile.to_widgets()
        # profile.to_notebook_iframe()
        profile.to_file('data_analysis_report.html')
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
        # return pd.get_dummies(x)

    @staticmethod
    def encode_categorical_y(y):
        le = LabelEncoder()
        return le.fit_transform(y)

    @staticmethod
    def normalize(x):
        scaler = StandardScaler()
        return scaler.fit_transform(x)

    @staticmethod
    def resample_data(x, y):
        smote = SMOTE()
        return smote.fit_resample(x, y)

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

    @staticmethod
    def apply_lda(x, y, n_components=None):
        lda = LinearDiscriminantAnalysis(n_components=n_components)
        return lda.fit_transform(x, y), lda

    @staticmethod
    def plot_lda(x, y, title):
        plt.figure(figsize=(8, 6))
        plt.scatter(x[:, 0], x[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
        plt.title(f'LDA Plot - {title}')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.colorbar(label='Label')
        plt.show()
        return None


class EvaluationMetrics:

    def confusion_matrix(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        print('Confusion Matrix:\n', cm)
        print('True Positive:', cm[1][1])
        print('False Positive:', cm[0][1])
        print('False Negative:', cm[1][0])
        print('True Negative:', cm[0][0])
        return None
    
    def plot_confusion_matrix(y_true, y_pred, title):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - {title}')
        plt.show()
    
    def classification_report(y_true, y_pred):
        cr = classification_report(y_true, y_pred=y_pred)
        print('Classification Report:\n', cr)
        return None
    
    def plot_roc_curve(y_true, y_pred, title):
        
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        auc_score = roc_auc_score(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], color='red', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {title}')
        plt.legend(loc='lower right')
        plt.grid()
        plt.show()
        return None
    
    def plot_precision_recall_curve(y_true, y_pred, title):
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', label='Precision-Recall curve')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {title}')
        plt.legend(loc='lower right')
        plt.grid()
        plt.show()
        return None

    def roc_auc_score(y_true, y_pred):
        print("ROC AUC Score: ", roc_auc_score(y_true, y_pred), '\n\n')
        return None
    
    def accuracy(y_true, y_pred):
        print('Accuracy Score: ', accuracy_score(y_true, y_pred))
        return None
    
    def silhouette_score(x, labels):
        if len(set(labels)) > 1:
            score = silhouette_score(x, labels)
        else:
            score = -1
        return score
    
    def calinski_harabasz_score(x, labels):
        if len(set(labels)) > 1:
            score = calinski_harabasz_score(x, labels)
        else:
            score = -1
        return score
    
    def davies_bouldin_score(x, labels):
        if len(set(labels)) > 1:
            score = davies_bouldin_score(x, labels)
        else:
            score = -1
        return score
    
    def plot_clusters(x, labels, title):
        plt.figure(figsize=(8, 6))
        plt.scatter(x[:, 0], x[:, 1], c=labels, cmap='viridis', edgecolor='k', s=50)
        plt.title(f'Cluster Plot - {title}')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.colorbar(label='Cluster Label')
        plt.show()

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

    def kMeans(x):

        EvaluationMetrics.elbow_method(x)

        kmeans = KMeans(n_clusters=2)
        param_grid = {
            'n_clusters': range(2, 11),
            'max_iter': [300, 500, 1000]
        }
        
        grid_search = GridSearchCV(kmeans, param_grid, cv=5)
        grid_search.fit(x)
        
        best_kmeans = grid_search.best_estimator_
        best_kmeans.fit(x)

        cluster_labels_kmeans = best_kmeans.labels_

        num_labels = len(np.unique(cluster_labels_kmeans))
        print(f"Number of clusters formed: {num_labels}")

        print(f"The labels assigned by KMeans are: {cluster_labels_kmeans}")

        EvaluationMetrics.plot_clusters(x, cluster_labels_kmeans, 'KMeans')

        return cluster_labels_kmeans


    def agglomerative_clustering(x):

        agg = AgglomerativeClustering(n_clusters=2)
        param_grid = {
            'n_clusters': range(2, 11)
        }
        
        grid_search = GridSearchCV(agg, param_grid, scoring=make_scorer(EvaluationMetrics.silhouette_score), cv=5)
        grid_search.fit(x)
        
        best_agg = grid_search.best_estimator_
        best_agg.fit(x)

        cluster_labels_agg = best_agg.labels_

        num_labels = len(np.unique(cluster_labels_agg))
        print(f"Number of clusters formed: {num_labels}")

        print(f"The labels assigned by Agglomerative Clustering are: {cluster_labels_agg}")

        EvaluationMetrics.plot_clusters(x, cluster_labels_agg, 'Agglomerative Clustering')

        return cluster_labels_agg
    

    def spectral_clustering(x):

        spectral = SpectralClustering(n_clusters=2)
        param_grid = {
            'n_clusters': range(2, 11)
        }
        
        grid_search = GridSearchCV(spectral, param_grid, scoring=make_scorer(EvaluationMetrics.silhouette_score), cv=5)
        grid_search.fit(x)
        
        best_spectral = grid_search.best_estimator_
        best_spectral.fit(x)

        cluster_labels_spectral = best_spectral.labels_

        num_labels = len(np.unique(cluster_labels_spectral))
        print(f"Number of clusters formed: {num_labels}")

        print(f"The labels assigned by Spectral Clustering are: {cluster_labels_spectral}")

        EvaluationMetrics.plot_clusters(x, cluster_labels_spectral, 'Spectral Clustering')

        return cluster_labels_spectral
    

    def gaussian_mixture(x):

        gmm = GaussianMixture(n_components=2)
        param_grid = {
            'n_components': range(2, 11)
        }
        
        grid_search = GridSearchCV(gmm, param_grid, cv=5)
        grid_search.fit(x)
        
        best_gmm = grid_search.best_estimator_
        best_gmm.fit(x)

        cluster_labels_gmm = best_gmm.predict(x)

        num_labels = len(np.unique(cluster_labels_gmm))
        print(f"Number of clusters formed: {num_labels}")

        print(f"The labels assigned by Gaussian Mixture Model are: {cluster_labels_gmm}")

        EvaluationMetrics.plot_clusters(x, cluster_labels_gmm, 'Gaussian Mixture Model')

        return cluster_labels_gmm

    
    def dbScan(x):

        dbscan = DBSCAN(eps=0.5, min_samples=5)
        cluster_labels_dbscan = dbscan.fit_predict(x)

        num_labels = len(np.unique(cluster_labels_dbscan))
        print(f"Number of clusters formed: {num_labels}")

        print(f"The labels assigned by DBSCAN are: {cluster_labels_dbscan}")

        EvaluationMetrics.plot_clusters(x, cluster_labels_dbscan, 'DBSCAN')

        return cluster_labels_dbscan
    

    def clustering_report(x, y=None):

        kmeans_labels = Clustering.kMeans(x)
        agg_labels = Clustering.agglomerative_clustering(x)
        spectral_labels = Clustering.spectral_clustering(x)
        gmm_labels = Clustering.gaussian_mixture(x)

        if (y is not None):
 
            EvaluationMetrics.confusion_matrix(y, kmeans_labels)
            EvaluationMetrics.plot_confusion_matrix(y, kmeans_labels, 'KMeans')
            EvaluationMetrics.classification_report(y, kmeans_labels)
            EvaluationMetrics.plot_roc_curve(y, kmeans_labels, 'KMeans')
            EvaluationMetrics.roc_auc_score(y, kmeans_labels)

            EvaluationMetrics.confusion_matrix(y, agg_labels)
            EvaluationMetrics.plot_confusion_matrix(y, agg_labels, 'Agglomerative Clustering')
            EvaluationMetrics.classification_report(y, agg_labels)
            EvaluationMetrics.plot_roc_curve(y, agg_labels, 'Agglomerative Clustering')
            EvaluationMetrics.roc_auc_score(y, agg_labels)
            
            EvaluationMetrics.confusion_matrix(y, spectral_labels)
            EvaluationMetrics.plot_confusion_matrix(y, spectral_labels, 'Spectral Clustering')
            EvaluationMetrics.classification_report(y, spectral_labels)
            EvaluationMetrics.plot_roc_curve(y, spectral_labels, 'Spectral Clustering')

            EvaluationMetrics.confusion_matrix(y, gmm_labels)
            EvaluationMetrics.plot_confusion_matrix(y, gmm_labels, 'Gaussian Mixture Model')
            EvaluationMetrics.classification_report(y, gmm_labels)
            EvaluationMetrics.plot_roc_curve(y, gmm_labels, 'Gaussian Mixture Model')

        scores = defaultdict(dict)

        # KMeans clustering
        kmeans_labels = Clustering.kMeans(x)
        scores['KMeans']['silhouette'] = EvaluationMetrics.silhouette_score(x, kmeans_labels)
        scores['KMeans']['calinski_harabasz'] = EvaluationMetrics.calinski_harabasz_score(x, kmeans_labels)
        scores['KMeans']['davies_bouldin'] = EvaluationMetrics.davies_bouldin_score(x, kmeans_labels)
        
        # Agglomerative clustering
        agg_labels = Clustering.agglomerative_clustering(x)
        scores['Agglomerative']['silhouette'] = EvaluationMetrics.silhouette_score(x, agg_labels)
        scores['Agglomerative']['calinski_harabasz'] = EvaluationMetrics.calinski_harabasz_score(x, agg_labels)
        scores['Agglomerative']['davies_bouldin'] = EvaluationMetrics.davies_bouldin_score(x, agg_labels)

        # Spectral clustering
        spectral_labels = Clustering.spectral_clustering(x)
        scores['Spectral']['silhouette'] = EvaluationMetrics.silhouette_score(x, spectral_labels)
        scores['Spectral']['calinski_harabasz'] = EvaluationMetrics.calinski_harabasz_score(x, spectral_labels)
        scores['Spectral']['davies_bouldin'] = EvaluationMetrics.davies_bouldin_score(x, spectral_labels)

        # Gaussian Mixture Model
        gmm_labels = Clustering.gaussian_mixture(x)
        scores['GMM']['silhouette'] = EvaluationMetrics.silhouette_score(x, gmm_labels)
        scores['GMM']['calinski_harabasz'] = EvaluationMetrics.calinski_harabasz_score(x, gmm_labels)
        scores['GMM']['davies_bouldin'] = EvaluationMetrics.davies_bouldin_score(x, gmm_labels)

        print(scores)

        best_model = max(scores, key=lambda x: scores[x]['silhouette'])

        return best_model, scores


def main() -> None:

    df = read_csv("./mall_customers.csv")

    df.drop('CustomerID', axis=1, inplace=True)
    
    PreProcessing.data_description(df)
    df = PreProcessing.drop_na_columns(df)
    
    x = df.iloc[:, :-1].values
    # y = df.iloc[:, -1].values

    x = PreProcessing.encode_categorical_x(x)
    # print(x)

    # y = PreProcessing.encode_categorical_y(y)

    df = PreProcessing.impute_missing_values(x)
    # x, y = PreProcessing.resample_data(x, y)
    x = PreProcessing.normalize(x)

    PreProcessing.data_profiling(x)

    # PreProcessing.plot_labels_distribution(y)

    best_model, scores = Clustering.clustering_report(x, y=None)
    print(f"Best Model: {best_model}")

    # x_pca, pca = PreProcessing.apply_pca(x, n_components=2)
    # PreProcessing.plot_pca(x_pca, y, 'PCA')

    # best_pca_model, pca_scores = Clustering.clustering_report(x_pca, y=None)
    # print(f"Best Model: {best_model}")

    
    # x_lda, lda = PreProcessing.apply_lda(x, y, n_components=2)
    # PreProcessing.plot_lda(x_lda, y, 'LDA')

    # best_lda_model, lda_scores = Clustering.clustering_report(x_lda, y=None)
    # print(f"Best Model: {best_model}")

    return None


if '__main__' == __name__:
    main()