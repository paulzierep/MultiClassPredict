import argparse
import copy
import itertools
import warnings

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import NearMiss, RandomUnderSampler
from lightgbm import LGBMClassifier
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import (
    SelectFromModel,
    SelectKBest,
    VarianceThreshold,
    f_classif,
)
from sklearn.metrics import (
    auc,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, label_binarize
from sklearn.svm import SVC

# from tabpfn import TabPFNClassifier
# from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier
from xgboost import XGBClassifier

# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)


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
        scoring="roc_auc_ovr",
    )
    y_score = cross_val_predict(ovr_clf, X, y, cv=outer_cv, method="predict_proba")

    # Calculate AUC for each class
    y_bin = LabelBinarizer().fit_transform(y)
    ovr_auc = roc_auc_score(y_bin, y_score, multi_class="ovr", average=None)
    for idx, auc_val in enumerate(ovr_auc):
        print(f"AUC for class '{class_names[idx]}': {auc_val:.4f}")
        results[f"{class_names[idx]} vs Rest - AUC"] = auc_val

    # Calculate precision, recall, F1, and MCC for each class in OvR
    y_pred_ovr = np.argmax(y_score, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, y_pred_ovr, average=None
    )
    for idx in range(len(class_names)):
        results[f"{class_names[idx]} vs Rest - Precision"] = precision[idx]
        results[f"{class_names[idx]} vs Rest - Recall"] = recall[idx]
        results[f"{class_names[idx]} vs Rest - F1"] = f1[idx]

        # Calculate MCC for each class in OvR
        mcc = matthews_corrcoef(y, (y_pred_ovr == idx).astype(int))
        results[f"{class_names[idx]} vs Rest - MCC"] = mcc

    # Calculate macro AUC, precision, recall, F1, and MCC for OvR
    macro_ovr_auc = roc_auc_score(y_bin, y_score, multi_class="ovr", average="macro")
    macro_ovr_precision = np.mean(precision)
    macro_ovr_recall = np.mean(recall)
    macro_ovr_f1 = np.mean(f1)
    macro_ovr_mcc = np.mean(
        [
            matthews_corrcoef(y, (y_pred_ovr == idx).astype(int))
            for idx in range(len(class_names))
        ]
    )

    results["OvR Macro AUC"] = macro_ovr_auc
    results["OvR Macro Precision"] = macro_ovr_precision
    results["OvR Macro Recall"] = macro_ovr_recall
    results["OvR Macro F1"] = macro_ovr_f1
    results["OvR Macro MCC"] = macro_ovr_mcc

    print(f"Macro AUC (OvR): {macro_ovr_auc:.4f}")
    print(f"Macro Precision (OvR): {macro_ovr_precision:.4f}")
    print(f"Macro Recall (OvR): {macro_ovr_recall:.4f}")
    print(f"Macro F1 (OvR): {macro_ovr_f1:.4f}")
    print(f"Macro MCC (OvR): {macro_ovr_mcc:.4f}")

    ####################
    # One-vs-One Classification
    ####################
    print("Performing One vs One classification")
    ovo_auc = {}
    ovo_precision = {}
    ovo_recall = {}
    ovo_f1 = {}
    ovo_mcc = {}
    class_pairs = split_classes(X, y)

    for (c1, c2), (X_subset, y_subset) in class_pairs.items():
        ovo_clf = GridSearchCV(
            estimator=base_clf,
            param_grid={k.replace("estimator__", ""): v for k, v in p_grid.items()},
            cv=inner_cv,
            scoring="roc_auc",
        )
        y_score = cross_val_predict(
            ovo_clf, X_subset, y_subset, cv=outer_cv, method="predict_proba"
        )
        y_binary = (y_subset == c2).astype(int)
        fpr, tpr, _ = roc_curve(y_binary, y_score[:, 1])
        auc_val = auc(fpr, tpr)

        # Compute precision, recall, F1, and MCC for each class pair (OvO)
        y_pred_ovo = np.argmax(y_score, axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_binary, y_pred_ovo, average="binary"
        )

        # MCC for each class pair (OvO)
        mcc = matthews_corrcoef(y_binary, y_pred_ovo)

        results[
            f"{le.inverse_transform([c1])[0]} vs {le.inverse_transform([c2])[0]} - AUC"
        ] = auc_val
        results[
            f"{le.inverse_transform([c1])[0]} vs {le.inverse_transform([c2])[0]} - Precision"
        ] = precision
        results[
            f"{le.inverse_transform([c1])[0]} vs {le.inverse_transform([c2])[0]} - Recall"
        ] = recall
        results[
            f"{le.inverse_transform([c1])[0]} vs {le.inverse_transform([c2])[0]} - F1"
        ] = f1
        results[
            f"{le.inverse_transform([c1])[0]} vs {le.inverse_transform([c2])[0]} - MCC"
        ] = mcc

        ovo_auc[(c1, c2)] = auc_val
        ovo_precision[(c1, c2)] = precision
        ovo_recall[(c1, c2)] = recall
        ovo_f1[(c1, c2)] = f1
        ovo_mcc[(c1, c2)] = mcc

    # Calculate macro AUC, precision, recall, F1, and MCC for OvO
    macro_ovo_auc = np.mean(
        list(ovo_auc.values())
    )  # Macro: Average AUC over all class pairs
    macro_ovo_precision = np.mean(list(ovo_precision.values()))
    macro_ovo_recall = np.mean(list(ovo_recall.values()))
    macro_ovo_f1 = np.mean(list(ovo_f1.values()))
    macro_ovo_mcc = np.mean(list(ovo_mcc.values()))

    results["OvO Macro AUC"] = macro_ovo_auc
    results["OvO Macro Precision"] = macro_ovo_precision
    results["OvO Macro Recall"] = macro_ovo_recall
    results["OvO Macro F1"] = macro_ovo_f1
    results["OvO Macro MCC"] = macro_ovo_mcc

    print(f"Macro AUC (OvO): {macro_ovo_auc:.4f}")
    print(f"Macro Precision (OvO): {macro_ovo_precision:.4f}")
    print(f"Macro Recall (OvO): {macro_ovo_recall:.4f}")
    print(f"Macro F1 (OvO): {macro_ovo_f1:.4f}")
    print(f"Macro MCC (OvO): {macro_ovo_mcc:.4f}")

    return results


def repeat_clf(n_seeds, ks, X, y, model, sampling_strategy):

    print(ks)
    print(n_seeds)

    # Define sampling strategies
    sampling_strategies = {
        "No Sampling": None,
        "Random OverSampling": RandomOverSampler(random_state=42),
        "SMOTE": SMOTE(random_state=42),
        "Random UnderSampling": RandomUnderSampler(random_state=42),
        "NearMiss (v1)": NearMiss(version=1),
        "NearMiss (v2)": NearMiss(version=2),
        "NearMiss (v3)": NearMiss(version=3),
    }

    # If the selected strategy is not in the dictionary, use "No Sampling"
    sampler = sampling_strategies.get(sampling_strategy, None)

    seed_results = {}

    for seed in range(n_seeds):

        ks_results = {}
        for k in ks:

            print(f"CV for seed {seed} and {k} features")

            # Create a Random Forest Classifier
            rf = RandomForestClassifier(random_state=seed)

            # Create a SelectFromModel using the Random Forest Classifier
            selector = SelectFromModel(rf, max_features=k)

            if model == "rf":
                ml_model = rf
                ml_model_grid = {
                    "estimator__classification__n_estimators": [
                        100
                    ],  # Number of trees in the forest
                    "estimator__classification__max_features": [
                        "sqrt"
                    ],  # Feature selection strategy
                    "estimator__classification__criterion": [
                        "entropy"
                    ],  # Split criterion
                    "estimator__classification__min_samples_leaf": [
                        3
                    ],  # Minimum samples per leaf
                }
            elif model == "xgb":
                ml_model = XGBClassifier(
                    use_label_encoder=False, eval_metric="logloss", random_state=seed
                )
                ml_model_grid = {
                    "estimator__classification__n_estimators": [100],
                    "estimator__classification__gamma": [0],
                    "estimator__classification__max_depth": [6],
                }
            elif model == "etc":
                ml_model = ExtraTreesClassifier(random_state=seed)
                ml_model_grid = {
                    "estimator__classification__n_estimators": [100],
                }
            elif model == "lgbm":
                ml_model = LGBMClassifier(random_state=seed, verbose=-1)
                ml_model_grid = {
                    "estimator__classification__n_estimators": [100],
                }

            # If there is a sampler, include it in the pipeline
            steps = []
            if sampler:
                steps.append(("sampling", sampler))
            steps.append(("feature_selection", selector))
            steps.append(("classification", ml_model))

            # Create a pipeline with feature selection, sampling, and classification
            pipeline = Pipeline(steps=steps)

            ###########################

            # Run the classification with the sampling strategy
            results = ovo_and_ova_multiclass_auc(
                X, y, pipeline, ml_model_grid, random_state=seed
            )

            print(results)

            ks_results[k] = {
                "results": results,
                "Dataset": X,
                "Model": model,
                "Sampling_Strategy": sampling_strategy,
            }


        seed_results[seed] = copy.copy(ks_results)

    return seed_results


def store_results(seed_results, output):

    # Flatten the nested dictionary into a DataFrame
    '''df = pd.DataFrame(
        {
            (outer_key, inner_key): values
            for outer_key, inner_dict in seed_results.items()
            for inner_key, values in inner_dict.items()
        }
    ).T

    # '''
    
    final_results = []
    
    for seed, ks_results in seed_results.items():
        for k, result_info in ks_results.items():
            result = result_info["results"]
            model = result_info["Model"]
            sampling_strategy = result_info["Sampling_Strategy"]
            dataset=result_info["Dataset"]
            
            # Collect all relevant information in a list
            final_results.append({
                "Seed": seed,
                "Features (k)": k,
                "Dataset":dataset,
                "Model": model,
                "Sampling Strategy": sampling_strategy,
                "Result": result,  # Assuming 'result' contains the results of your classification
            })
    df = pd.DataFrame(final_results)

    #Set multi-level index names for clarity
    df.set_index(["Seed", "Features (k)", "Dataset", "Model", "Sampling Strategy"], inplace=True)

    df.index.names = ["Seed", "Features (k)","Dataset","Model","Sampling Strategy"]
    # Display the DataFrame
    df = df.reset_index()

    df.to_csv(output, mode='a', header=not os.path.exists(output))

    print(df)


def run_classification(X, y, ks, n_seeds,output,model, sampling_strategy):

    # Ensure ks does not exceed the number of columns in X
    max_features = len(X.columns)
    ks = [k for k in ks if k <= max_features]
    if max_features not in ks:
        ks.append(max_features)

    seed_results = repeat_clf(n_seeds, ks, X, y, model, sampling_strategy)
    store_results(seed_results, output)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run Classification Model")

    parser.add_argument("--X", type=str, required=True, help="path to X")
    parser.add_argument("--y", type=str, required=True, help="path to y")
    parser.add_argument(
        "--ks", type=int, nargs="+", required=True, help="list of values of k"
    )
    parser.add_argument("--n_seeds", type=int, default=2, help="number of seeds")
    parser.add_argument("--model", type=str, required=True, help="choose model :['rf', 'XGB', 'ETC', 'lgbm' ]")
    parser.add_argument("--sampling_strategy", type=str, required=True, help="choose sampling strategy: ['No Sampling','Random OverSampling','SMOTE','Random UnderSampling','NearMiss (v1)','NearMiss (v2)','NearMiss (v3)']")
    
    args = parser.parse_args()

    # reading str file paths
    X = pd.read_csv(args.X)
    y = pd.read_csv(args.y)

    # flattening y into 1D array
    y = y["target"].values.ravel()
    result_path="results/appended_results.csv"

    run_classification(X, y, args.ks, args.n_seeds,result_path,args.model, args.sampling_strategy)
