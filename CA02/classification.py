from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier

cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])


class KNN: 
    def __init__(self, k=1) -> None:
        self.k = k
    
    def fit(self, x, y):
        # Fit the training data
        self.X_train = x
        self.y_train = y
    
    def predict(self, X):
        # Make predictions on the test data
        predictions = [self._predict(x) for x in X]
        return predictions
    
    def euclidean_distance(self, x1, x2):
        # Compute the Euclidean distance between two vectors
        return np.sqrt(np.sum((x1 - x2)**2))
    
    def _predict(self, x):
        # Compute distances between x and all examples in the training set
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]

        # Get Closets K samples
        k_indices = np.argsort(distances)[:self.k]

        # Extract the labels of the K nearest neighbor training samples
        k_nearest_labels = [self.y_train.iloc[i] for i in k_indices]

        # Make predictions based on the closest k samples
        most_common = np.bincount(k_nearest_labels).argmax()

        # most_common = Counter(k_nearest_labels).most_common(1)[0][0]
        
        return most_common
    

class PreProcessing:

    def drop_na_columns(data):
        # Drop columns where all values are NaN
        return data.dropna(axis=1, how='all')
    
    def map_diagnosis(diagnosis):
        # Map diagnosis to 0 and 1
        return diagnosis.map({'B': 0, 'M': 1})
    
    def impute_missing_values(data):
        # Impute missing values with mean
        imputer = SimpleImputer(strategy='mean')
        imputer.fit(data)
        return imputer.transform(data)
    
    def resample_data(x, y):
        # Resample data using SMOTE
        smote = SMOTE()
        return smote.fit_resample(x, y)
    
    def plot_labels_distribution(y):
        # Count the occurrences of each diagnosis
        counts = [sum(y == 0), sum(y == 1)]

        # Plot the distribution of labels using a bar chart
        plt.figure(figsize=(8, 6))
        plt.bar(['Benign', 'Malignant'], counts, color=['blue', 'red'], edgecolor='black')
        plt.xlabel('Diagnosis')
        plt.ylabel('Count')
        plt.title('Diagnosis Distribution')
        plt.show()

        return None
    

class EvaluationMetrics:

    def accuracy(y_true, y_pred):
        # Calculate the accuracy of the model
        print('Accuracy Score: ', accuracy_score(y_true, y_pred))
        return None
    
    def confusion_matrix(y_true, y_pred):
        # Calculate the confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        print('Confusion Matrix:\n', cm)
        print('True Positive:', cm[0][0])
        print('False Positive:', cm[0][1])
        print('False Negative:', cm[1][0])
        print('True Negative:', cm[1][1])
        return None
    
    def classification_report(y_true, y_pred):
        # Generate a classification report
        cr = classification_report(y_true, y_pred=y_pred)
        print('Classification Report:\n', cr)
        return None

    def roc_auc_score(y_true, y_pred):
    # Calculate the AUC-ROC score
        print("ROC AUC Score: ", roc_auc_score(y_true, y_pred))
        return None
    
    def plot_roc_curve(y_true, y_pred):
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)

        # Calculate AUC score
        auc_score = roc_auc_score(y_true, y_pred)

        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.grid()
        plt.show()
        return None


def main() -> None:
    # Load the data
    data = read_csv('./dataset/breast_cancer.csv')
    
    # Drop columns where all values are NaN
    data = PreProcessing.drop_na_columns(data)
    
    # Map diagnosis to 0 and 1
    y = PreProcessing.map_diagnosis(data['diagnosis'])
    print(Counter(y))
    PreProcessing.plot_labels_distribution(y)
    
    # Drop diagnosis column
    x = data.drop('diagnosis', axis=1)
    
    # Impute missing values with mean
    x = PreProcessing.impute_missing_values(x)
    
    # Resample data using SMOTE
    x, y = PreProcessing.resample_data(x, y)
    print(Counter(y))
    
    # Plot the distribution of labels
    PreProcessing.plot_labels_distribution(y)
    
    # Split the data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    
    # Create a KNN classifier
    knn = KNN(k=3)
    
    # Fit the training data
    knn.fit(x_train, y_train)
    
    # Create and evaluate custom KNN classifier
    custom_knn = KNN(k=3)
    custom_knn.fit(x_train, y_train)
    y_pred_custom = custom_knn.predict(x_test)
    
    print("Custom KNN Classifier Metrics:")
    EvaluationMetrics.accuracy(y_test, y_pred_custom)
    EvaluationMetrics.confusion_matrix(y_test, y_pred_custom)
    EvaluationMetrics.classification_report(y_test, y_pred_custom)
    EvaluationMetrics.roc_auc_score(y_test, y_pred_custom)
    EvaluationMetrics.plot_roc_curve(y_test, y_pred_custom)


    # Create and evaluate sklearn KNN classifier
    sklearn_knn = KNeighborsClassifier(n_neighbors=3)
    sklearn_knn.fit(x_train, y_train)
    y_pred_sklearn = sklearn_knn.predict(x_test)
    
    print("\nSklearn KNN Classifier Metrics:")
    EvaluationMetrics.accuracy(y_test, y_pred_sklearn)
    EvaluationMetrics.confusion_matrix(y_test, y_pred_sklearn)
    EvaluationMetrics.classification_report(y_test, y_pred_sklearn)
    EvaluationMetrics.roc_auc_score(y_test, y_pred_sklearn)
    EvaluationMetrics.plot_roc_curve(y_test, y_pred_sklearn)


    return None
    

if __name__ == '__main__':
    main()