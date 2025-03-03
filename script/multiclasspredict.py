import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.metrics import (roc_curve, auc, roc_auc_score, 
                             precision_recall_fscore_support)
from sklearn.model_selection import (train_test_split, GridSearchCV, 
                                     KFold, StratifiedKFold, cross_val_score, 
                                     cross_val_predict)
from sklearn.preprocessing import label_binarize, LabelEncoder, LabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (SelectKBest, f_classif, VarianceThreshold, 
                                       SelectFromModel)
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import load_iris
import itertools

from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import copy

from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.model_selection import StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
import itertools

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import argparse




def split_classes(X, y):
    return {
        (c1, c2): (X[(y == c1) | (y == c2)], y[(y == c1) | (y == c2)])
        for c1, c2 in itertools.combinations(np.unique(y), 2)
    }

def ovo_and_ova_multiclass_auc(X, y, base_clf, p_grid, random_state):
    
    results = {}
    le = LabelEncoder()
    y = le.fit_transform(y)
    class_names = le.classes_

    # Stratified K-Folds
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    ####################
    # One-vs-Rest Classification
    ####################
    print("Performing One vs Rest classification")
    ovr_clf = GridSearchCV(
        estimator=OneVsRestClassifier(base_clf),
        param_grid=p_grid,
        cv=inner_cv,
        scoring="roc_auc_ovr"
    )
    y_score = cross_val_predict(ovr_clf, X, y, cv=outer_cv, method="predict_proba")

    # Calculate AUC for each class
    y_bin = LabelBinarizer().fit_transform(y)
    ovr_auc = roc_auc_score(y_bin, y_score, multi_class="ovr", average=None)
    for idx, auc_val in enumerate(ovr_auc):
        print(f"AUC for class '{class_names[idx]}': {auc_val:.4f}")
        results[f"{class_names[idx]} vs Rest"] = auc_val

    # Calculate macro and micro AUC for OvR
    macro_ovr_auc = roc_auc_score(y_bin, y_score, multi_class="ovr", average="macro")
    # micro_ovr_auc = roc_auc_score(y_bin, y_score, multi_class="ovr", average="micro")
    results["OvR Macro AUC"] = macro_ovr_auc
    print(f"Macro AUC (OvR): {macro_ovr_auc:.4f}")

    ####################
    # One-vs-One Classification
    ####################
    print("Performing One vs One classification")
    ovo_auc = {}
    class_pairs = split_classes(X, y)

    for (c1, c2), (X_subset, y_subset) in class_pairs.items():
        ovo_clf = GridSearchCV(
            estimator=base_clf,
            param_grid={k.replace("estimator__", ""): v for k, v in p_grid.items()},
            cv=inner_cv,
            scoring="roc_auc"
        )
        y_score = cross_val_predict(ovo_clf, X_subset, y_subset, cv=outer_cv, method="predict_proba")
        y_binary = (y_subset == c2).astype(int)
        fpr, tpr, _ = roc_curve(y_binary, y_score[:, 1])
        auc_val = auc(fpr, tpr)

        # Decode labels
        results[f"{le.inverse_transform([c1])[0]} vs {le.inverse_transform([c2])[0]}"] = auc_val
        ovo_auc[(c1, c2)] = auc_val

    # Calculate macro and micro AUC for OvO
    macro_ovo_auc = np.mean(list(ovo_auc.values()))  # Macro: Average AUC over all class pairs
    y_scores = cross_val_predict(base_clf, X, y, cv=outer_cv, method="predict_proba")
    results["OvO Macro AUC"] = macro_ovo_auc
    print(f"Macro AUC (OvO): {macro_ovo_auc:.4f}")

    return results

def repeat_clf(n_seeds, ks, X, y):

    print(ks)
    print(n_seeds)

    seed_results = {}

    for seed in range(n_seeds):

        ks_results = {}
        for k in ks:

            print(f"CV for seed {seed} and {k} features")

            # Create a Random Forest Classifier
            rf = RandomForestClassifier(random_state=seed)

            # Create a SelectFromModel using the Random Forest Classifier
            selector = SelectFromModel(rf, max_features = k)

            # Create a pipeline with feature selection and classification
            pipeline = Pipeline(steps=[
                ('feature_selection', selector),
                ('classification', rf)
            ])

            # Parameter grid for RandomForestClassifier
            p_grid = {
                "estimator__classification__n_estimators": [100],          # Number of trees in the forest
                "estimator__classification__max_features": ["sqrt"],       # Feature selection strategy
                "estimator__classification__criterion": ["entropy"],       # Split criterion
                "estimator__classification__min_samples_leaf": [3],        # Minimum samples per leaf
            }

            ###########################
                
            results = ovo_and_ova_multiclass_auc(X,y,pipeline, p_grid, random_state=seed)

            print(results)

            ks_results[k] = results

        seed_results[seed] = copy.copy(ks_results)
    
    return(seed_results)

def store_results(seed_results, output):

    # Flatten the nested dictionary into a DataFrame
    df = pd.DataFrame(
        {(outer_key, inner_key): values for outer_key, inner_dict in seed_results.items() for inner_key, values in inner_dict.items()}
    ).T

    # Set multi-level index names for clarity
    df.index.names = ['Seed', 'Features (k)']

    # Display the DataFrame
    df = df.reset_index()

    df.to_csv(output)
    print(df)
    

def run_classification(X, y, ks, n_seeds, output):

    # Ensure ks does not exceed the number of columns in X
    max_features = len(X.columns)
    ks = [k for k in ks if k <= max_features]
    if max_features not in ks:
        ks.append(max_features)

    seed_results = repeat_clf(n_seeds, ks, X, y)
    store_results(seed_results, output)



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run Classification Model")
    
    parser.add_argument("--X", type=str, required=True, help="path to X")
    parser.add_argument("--y", type=str, required=True, help="path to y")
    parser.add_argument("--ks", type=int, nargs='+', required=True, help="list of values of k")
    parser.add_argument("--n_seeds", type=int, default=2, help="number of seeds")

    args = parser.parse_args()
    
    #reading str file paths
    X = pd.read_csv(args.X) 
    y = pd.read_csv(args.y)
    
    #flattening y into 1D array
    y = y["target"].values.ravel()
    
    run_classification(X,y, args.ks, args.n_seeds, "test_clf.csv")









