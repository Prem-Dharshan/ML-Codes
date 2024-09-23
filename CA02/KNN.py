import numpy as np


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