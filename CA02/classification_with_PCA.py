from collections import Counter
import matplotlib.pyplot as plt
from pandas import read_csv, DataFrame
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split
from matplotlib.colors import ListedColormap
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    

class PreProcessing:

    def normalize(x):
        # Standardize features
        scaler = StandardScaler()
        return scaler.fit_transform(x), scaler
    
    
    def apply_pca(x, n_components=None):
        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=n_components)
        x_pca = pca.fit_transform(x)
        explained_variance = pca.explained_variance_ratio_
        print(f"Explained variance by each component: {explained_variance}")
        print(f"Total explained variance: {sum(explained_variance)}")
        return x_pca, pca

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
        # Generate a classification report
        cr = classification_report(y_true, y_pred=y_pred)
        print('Classification Report:\n', cr)
        return None

    def roc_auc_score(y_true, y_pred):
        # Calculate the AUC-ROC score
        print("ROC AUC Score: ", roc_auc_score(y_true, y_pred), '\n\n')
        return None
    
    def plot_roc_curve(y_true, y_pred, title):
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
        plt.title(f'ROC Curve - {title}')
        plt.legend(loc='lower right')
        plt.grid()
        plt.show()
        return None


def main() -> None:
    # Load the data
    data = read_csv('./dataset/breast_cancer.csv')
    
    # Drop columns where all values are NaN
    data = PreProcessing.drop_na_columns(data)
    data.drop('id', axis=1, inplace=True)
    
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
    
    # Standardize features and get the scaler
    x, scaler = PreProcessing.normalize(x)

    # Define different numbers of components for PCA
    pca_components = [2, 5, 10, 20, 0.95]  # Last value is for explained variance ratio (95%)
    model_results = {}

    for n in pca_components:
        # Apply PCA
        if isinstance(n, float):
            x_pca, pca = PreProcessing.apply_pca(x, n_components=n)  # Retain 95% of variance
            print(f'\nApplying PCA with {n*100}% variance explained.')
        else:
            x_pca, pca = PreProcessing.apply_pca(x, n_components=n)  # n is the number of components
            print(f'\nApplying PCA with {n} components.')
        
        # Split the data into training and test sets
        x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.2)
        
        # Evaluate different models (KNN, SVC, Decision Tree, etc.)
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(x_train, y_train)
        y_pred_knn = knn.predict(x_test)
        y_pred_proba_knn = knn.predict_proba(x_test)[:, 1] # Probability of Malignant

        accuracy_knn = accuracy_score(y_test, y_pred_knn)
        auc_knn = roc_auc_score(y_test, y_pred_proba_knn)

        # Store results
        model_results[f'KNN - PCA {n}'] = {
            'accuracy': accuracy_knn,
            'auc': auc_knn
        }

        print(f"KNN Classifier with PCA {n}: Accuracy = {accuracy_knn:.4f}, AUC = {auc_knn:.4f}")
        
        # Similarly, apply and evaluate other classifiers like SVC, DecisionTree
        svc = SVC(kernel='linear', probability=True)
        svc.fit(x_train, y_train)
        y_pred_svc = svc.predict(x_test)
        y_pred_proba_svc = svc.predict_proba(x_test)[:, 1] # Probability of Malignant

        accuracy_svc = accuracy_score(y_test, y_pred_svc)
        auc_svc = roc_auc_score(y_test, y_pred_proba_svc)

        model_results[f'SVC - PCA {n}'] = {
            'accuracy': accuracy_svc,
            'auc': auc_svc
        }

        print(f"SVC Classifier with PCA {n}: Accuracy = {accuracy_svc:.4f}, AUC = {auc_svc:.4f}")

        # Decision Tree Classifier
        dtc = DecisionTreeClassifier()
        dtc.fit(x_train, y_train)
        y_pred_dtc = dtc.predict(x_test)
        y_pred_proba_dtc = dtc.predict_proba(x_test)[:, 1] # Probability of Malignant

        accuracy_dtc = accuracy_score(y_test, y_pred_dtc)
        auc_dtc = roc_auc_score(y_test, y_pred_proba_dtc)

        model_results[f'Decision Tree - PCA {n}'] = {
            'accuracy': accuracy_dtc,
            'auc': auc_dtc
        }

        print(f"Decision Tree Classifier with PCA {n}: Accuracy = {accuracy_dtc:.4f}, AUC = {auc_dtc:.4f}")

    # Summary of results
    print("\nSummary of Results with Different PCA Dimensions:")
    for model, results in model_results.items():
        print(f"{model} -> Accuracy: {results['accuracy']:.4f}, AUC: {results['auc']:.4f}")

if __name__ == '__main__':
    main()