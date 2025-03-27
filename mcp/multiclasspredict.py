import argparse
import copy
import itertools
import warnings
import os

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


def repeat_clf(n_seeds, ks, X, y, label, model, sampling_strategy):

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
                "Label": label,
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
            label=result_info["Label"]
            
            # Collect all relevant information in a list
            final_results.append({
                "Seed": seed,
                "Features (k)": k,
                "Label":label,
                "Model": model,
                "Sampling_Strategy": sampling_strategy,
                **result, 
            })
    df = pd.DataFrame(final_results)

    #Set multi-level index names for clarity
    df.set_index(["Seed", "Features (k)", "Label", "Model", "Sampling_Strategy"], inplace=True)

    df.index.names = ["Seed", "Features (k)","Label","Model","Sampling_Strategy"]
    # Display the DataFrame
    df = df.reset_index()

    df.to_csv(output, mode='a', header=not os.path.exists(output), index=False)

    print(df)


def run_classification(X, y, ks, n_seeds,output, label,model, sampling_strategy):

    # Ensure ks does not exceed the number of columns in X
    max_features = len(X.columns)
    ks = [k for k in ks if k <= max_features]
    if max_features not in ks:
        ks.append(max_features)

    seed_results = repeat_clf(n_seeds, ks, X, y, label,model, sampling_strategy)
    store_results(seed_results, output)



def plot_bar(output,ks,label,model,sampling_strategy):
    # Assuming `df` is your DataFrame
    df = pd.read_csv(output)
   
    # Resetting the index to simplify handling hierarchical index
    df = df.reset_index()
    
    # Filter the DataFrame to select only rows with required 'label' 
    df = df[(df['Label'] == label) & 
        (df['Model'] == model) & 
        (df['Sampling_Strategy'] == sampling_strategy)]


    #dropping columns as values irrelevant for metric
    df = df.drop(columns=['index','Unnamed: 0','Label','Model','Sampling_Strategy'], errors='ignore')
    #print(df)
    
    # Melting the dataframe to long format for seaborn compatibility
    df_melted = df.melt(
        id_vars=["Seed", "Features (k)"], 
        var_name="Metric", 
        value_name="Value"
    )
    df_melted = df_melted[df_melted["Metric"].str.contains("AUC", na=False)]

   

    print("\ndf_melted:\n",df_melted)
    
    # Create a new figure for each plot to prevent them from overlaying
    plt.figure(figsize=(16, 6))
    # Initialize the subplots
    fig, axes = plt.subplots(1, len(ks), figsize=(16, 6), sharey=True)
    
    # If only one subplot, make sure axes is iterable (a list of length 1)
    if len(ks) == 1:
        axes = [axes]
    
    # Iterate over the subsets ks
    for i, features in enumerate(ks):
        
        df_melted["Features (k)"] = df_melted["Features (k)"].astype(str)
        # Remove any quotes and strip spaces
        df_melted["Features (k)"] = df_melted["Features (k)"].str.replace(r"['\"]", "", regex=True)

        # Convert to numeric
        df_melted["Features (k)"] = pd.to_numeric(df_melted["Features (k)"], errors='coerce')
       
        subset = df_melted[df_melted["Features (k)"] == features]
        
        # Check if subset is empty
        if subset.empty:  
            print(f"Warning: No data for Features (k) = {features}")
            # Skip plotting for this subset
            continue  
    
        # Plot barplot for averages with error bars
        
        sns.barplot(
            data=subset, 
            x="Metric", 
            y="Value", 
            ax=axes[i], 
            ci="sd", 
            color="skyblue",
            estimator="mean",
           
        )
    
        # Overlay stripplot for individual seed values
        sns.stripplot(
            data=subset, 
            x="Metric", 
            y="Value", 
            ax=axes[i], 
            hue="Seed",
            dodge=True, 
            jitter=True, 
            alpha=0.7,
            
        )
    
        # Add a horizontal dashed red line for AUC = 0.5
        axes[i].axhline(0.5, color="red", linestyle="--", linewidth=1.5)
    
        # Set subplot title
        axes[i].set_title(f"Features (k): {features}, Model : {model}, sampling_strategy: {sampling_strategy}")
    
        # Rotate x-axis labels for clarity
        axes[i].tick_params(axis='x', rotation=90)
       
    
    
        
    # Set overall title and adjust layout
    fig.suptitle("Bar Plot with Individual Seed Values and AUC=0.5 Threshold")
    fig.tight_layout()
    plt.show()





def plot_all_modes(output, label, model, sampling_strategy):
    df = pd.read_csv(output, index_col = 0)

    # Filter the DataFrame to select only rows with required 'label' 
    df = df[(df['Label'] == label) & 
        (df['Model'] == model) & 
        (df['Sampling_Strategy'] == sampling_strategy)]

    
    #converting features to str to remove unnecessary data
    df["Features (k)"] = df["Features (k)"].astype(str)
    
    # Remove any quotes and strip spaces
    df["Features (k)"] = df["Features (k)"].str.replace(r"['\"]", "", regex=True)

    #removing columns as irrelevant to metric
    df = df.drop(columns=['index','Unnamed: 0','Label','Model','Sampling_Strategy'], errors='ignore')
    
    # Convert to numeric
    df["Features (k)"] = pd.to_numeric(df["Features (k)"], errors='coerce')
    
    # Group by 'Features (k)' and calculate mean and std for each metric
    metrics = df.columns[5:]  # All columns after 'Seed' and 'Features (k)'
    summary = df.groupby("Features (k)").agg(["mean", "std"])
    
    # Map 'Features (k)' to evenly spaced indices
    features = summary.index
    x_indices = range(len(features))
    print(df)
    # Separate OvR and OvO metrics
    ovr_metrics = ["0 vs Rest - AUC","1 vs Rest - AUC","2 vs Rest - AUC", "OvR Macro AUC"]
    ovo_metrics = ["0 vs 1 - AUC","0 vs 2 - AUC","1 vs 2 - AUC", "OvO Macro AUC"]
    
    # Find the y-axis range for all metrics
    y_min = max(0, min(summary[(metric, "mean")].min() for metric in metrics) - 0.1)
    y_max = max(summary[(metric, "mean")].max() for metric in metrics) + 0.1
    
    # Generate a colormap
    color_map = cm.get_cmap("tab10", len(ovr_metrics))  # Tab10 colormap for 4 colors

    plt.figure(figsize=(12, 8))
    # Plot OvR metrics
    fig_ovr, axes_ovr = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes_ovr = axes_ovr.flatten()
    
    for i, (ax, metric) in enumerate(zip(axes_ovr, ovr_metrics)):
        means = summary[(metric, "mean")]
        stds = summary[(metric, "std")]
        ax.errorbar(
            x_indices,
            means,
            yerr=stds,
            capsize=5,
            marker="o",
            color=color_map(i),
            label=metric,
        )
        ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, label="Random Prediction")
        ax.set_title(f"{metric}, model:{model}, sampling_strategy:{sampling_strategy}", fontsize=10)
        ax.set_ylabel("Metric Value", fontsize=8)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        ax.set_ylim(0, y_max)  # Apply the same y-axis range
    
    axes_ovr[-1].set_xticks(x_indices)
    axes_ovr[-1].set_xticklabels(features)
    axes_ovr[-1].set_xlabel("Features (k)", fontsize=10)
    
    plt.tight_layout()
    fig_ovr.suptitle("OvR Metrics", fontsize=14, y=1.02)
    plt.show()
    
    # Plot OvO metrics
    fig_ovo, axes_ovo = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes_ovo = axes_ovo.flatten()
    
    for i, (ax, metric) in enumerate(zip(axes_ovo, ovo_metrics)):
        means = summary[(metric, "mean")]
        stds = summary[(metric, "std")]
        ax.errorbar(
            x_indices,
            means,
            yerr=stds,
            capsize=5,
            marker="o",
            color=color_map(i),
            label=metric,
        )
        ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, label="Random Prediction")
        ax.set_title(f"{metric}, model: {model}, sampling_strategy: {sampling_strategy}", fontsize=10)
        ax.set_ylabel("Metric Value", fontsize=8)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        ax.set_ylim(y_min, y_max)  # Apply the same y-axis range
    
    axes_ovo[-1].set_xticks(x_indices)
    axes_ovo[-1].set_xticklabels(features)
    axes_ovo[-1].set_xlabel("Features (k)", fontsize=10)
    
    plt.tight_layout()
    fig_ovo.suptitle("OvO Metrics", fontsize=14, y=1.02)
    plt.show()

def plot_box(file_path,x_axis):
    
    df = pd.read_csv(file_path)
    
    # Convert "Features (k)" to numeric
    df["Features (k)"] = pd.to_numeric(df["Features (k)"], errors="coerce")
    
    # Separate data based on number of features
    df_10_features = df[df["Features (k)"] == 10]
    df_all_features = df[df["Features (k)"] != 10]  # Assuming "all features" is any value other than 10
    
    # Set up plotting style
    sns.set_theme(style="whitegrid")
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot OvR Macro AUC for 10 features
    sns.boxplot(data=df_10_features, x=x_axis, y="OvR Macro AUC", hue="Label", ax=axes[0, 0])
    axes[0, 0].set_title("OvR Macro AUC (10 Features)")
    
    # Plot OvO Macro AUC for 10 features
    sns.boxplot(data=df_10_features, x=x_axis, y="OvO Macro AUC", hue="Label", ax=axes[0, 1])
    axes[0, 1].set_title("OvO Macro AUC (10 Features)")
    
    # Plot OvR Macro AUC for all features
    sns.boxplot(data=df_all_features, x=x_axis, y="OvR Macro AUC", hue="Label", ax=axes[1, 0])
    axes[1, 0].set_title("OvR Macro AUC (All Features)")
    
    # Plot OvO Macro AUC for all features
    sns.boxplot(data=df_all_features, x=x_axis, y="OvO Macro AUC", hue="Label", ax=axes[1, 1])
    axes[1, 1].set_title("OvO Macro AUC (All Features)")
    
    # Adjust layout and save figure
    plt.tight_layout()
    plt.savefig("AUC_Comparison_10_vs_All_Features.png")
    plt.show()

def classwise_auc_plot(file_path,x_axis):
    #read df
    df = pd.read_csv(file_path)
    
    # Convert "Features (k)" to numeric
    df["Features (k)"] = pd.to_numeric(df["Features (k)"], errors="coerce")
    
    # Separate data based on number of features
    df_10_features = df[df["Features (k)"] == 10]
    df_all_features = df[df["Features (k)"] != 10]  # Assuming "all features" means not 10
    
    # Define class-wise AUC columns
    class_auc_columns = [
        "Negative control vs Rest - AUC",
        "Patient vs Rest - AUC",
        "Positive control vs Rest - AUC",
        "Negative control vs Patient - AUC",
        "Negative control vs Positive control - AUC",
        "Patient vs Positive control - AUC"
    ]
    
    # Set up plotting style
    sns.set_theme(style="whitegrid")
    
    # --- PLOT CLASS-WISE AUCs ---
    for col in class_auc_columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
        # Plot for 10 features
        sns.boxplot(data=df_10_features, x=x_axis, y=col, hue="Label", ax=axes[0])
        axes[0].set_title(f"{col} (10 Features)")
    
        # Plot for all features
        sns.boxplot(data=df_all_features, x=x_axis, y=col, hue="Label", ax=axes[1])
        axes[1].set_title(f"{col} (All Features)")
    
        # Adjust layout and save the figure
        plt.tight_layout()
        filename = col.replace(" ", "_").replace("-", "").replace("/", "_") + "_Comparison.png"
        plt.savefig(filename)
        plt.show()



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run Classification Model")

    parser.add_argument("--X", type=str, required=True, help="path to X")
    parser.add_argument("--y", type=str, required=True, help="path to y")
    parser.add_argument(
        "--ks", type=int, nargs="+", required=True, help="list of values of k"
    )
    parser.add_argument("--n_seeds", type=int, default=2, help="number of seeds")
    parser.add_argument("--label", type=str, required=True, help="add label for clarity")
    parser.add_argument("--model", type=str, required=True, help="choose model :['rf', 'XGB', 'ETC', 'lgbm' ]")
    parser.add_argument("--sampling_strategy", type=str, required=True, help="choose sampling strategy: ['No Sampling','Random OverSampling','SMOTE','Random UnderSampling','NearMiss (v1)','NearMiss (v2)','NearMiss (v3)']")
    
    args = parser.parse_args()

    # reading str file paths
    X = pd.read_csv(args.X)
    y = pd.read_csv(args.y)

    # flattening y into 1D array
    y = y["target"].values.ravel()
    result_path="results/appended_results.csv"

    run_classification(X, y, args.ks, args.n_seeds,result_path, args.label,args.model, args.sampling_strategy)
    bar_plot(result_path,args.ks)
    plot_all_modes(result_path)
