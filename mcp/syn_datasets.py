from sklearn.datasets import make_classification
import pandas as pd

# Generate a synthetic dataset
X, y = make_classification(
    n_samples=150,      # Number of samples
    n_features=1000,    # Total number of features
    n_informative=20,   # Number of informative features
    n_redundant=200,    # Number of redundant features
    n_classes=3,        # Number of target classes
    weights=[0.7, 0.2, 0.1],  # Proportion for each class
    random_state=42     # For reproducibility
)

# Convert to DataFrame
X = pd.DataFrame(X)
y = pd.Series(y, name="target")

# Display class distribution
print(y.value_counts())


