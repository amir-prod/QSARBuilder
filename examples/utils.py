import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis \
     import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef, make_scorer, r2_score
import sklearn.metrics as metrics
# from imblearn.over_sampling import RandomOverSampler
# from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import umap
import seaborn as sns


# model evaluation function
def Eval_Model(ytrue, ypred, folder_name=None, filename=None):
    precision = precision_score(ytrue, ypred)
    recall = recall_score(ytrue, ypred)
    f1 = f1_score(ytrue, ypred)
    matthews = matthews_corrcoef(ytrue, ypred)
    accuracy = accuracy_score(ytrue, ypred)
    tn, fp, fn, tp = confusion_matrix(ytrue, ypred).ravel()
    specificity = tn / (tn+fp)
    cm = confusion_matrix(ytrue, ypred)
    cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [0, 1])
    if folder_name:
        cm_display.plot()
        plt.savefig(f'{folder_name}/{filename}.png')
        plt.close()
        print(f'precision: {precision}, recall: {recall}, f1: {f1}, accuracy: {accuracy}, specificity: {specificity}, matthews; {matthews}')
    return precision, recall, f1, accuracy, specificity, matthews


# forward sequential feature selection funciton
def sfs_feature_selection(model, xtrain, xtest, ytrain, feature_no, scorer=make_scorer(matthews_corrcoef), fixed_feature=None, verbose=1):
    sfs = SFS(model, 
           k_features=feature_no, 
           forward=True, 
           floating=False, 
           verbose=verbose,
           scoring=scorer,
           cv=0,
           n_jobs=-1,
           fixed_features=fixed_feature)
    sfs = sfs.fit(xtrain.values, ytrain.values.ravel())
    Xtrain = xtrain.iloc[:, list(sfs.k_feature_idx_)]
    Xtest = xtest.iloc[:, list(sfs.k_feature_idx_)]

    return Xtrain, Xtest, sfs

# get the feature index from sfs1.subsets_ and use it to build each model
def build_each_model(X_train, y_train, X_test, y_test, model, sfs, scorer=make_scorer(matthews_corrcoef), classification=True):
    scorer_train = []
    scorer_test = []
    feature_index_dict = {
        "classifier" if classification else "regressor": [],
        "feature_idx": [],
        "scorer_train": [],
        "scorer_test": []
    }
    for i in sfs.subsets_:
        feature_idx = list(sfs.subsets_[i]['feature_idx'])
        X_train_subset = X_train.iloc[:, feature_idx]
        X_test_subset = X_test.iloc[:, feature_idx]
        model.fit(X_train_subset, y_train)
        y_pred_test = model.predict(X_test_subset)
        y_pred_train = model.predict(X_train_subset)
        feature_index_dict["classifier" if classification else "regressor"].append(model.__class__.__name__)
        feature_index_dict["feature_idx"].append(feature_idx)
        if classification:
            feature_index_dict["scorer_train"].append(matthews_corrcoef(y_train, y_pred_train))
            feature_index_dict["scorer_test"].append(matthews_corrcoef(y_test, y_pred_test))
        else:
            feature_index_dict["scorer_train"].append(r2_score(y_train, y_pred_train))
            feature_index_dict["scorer_test"].append(r2_score(y_test, y_pred_test))
    return feature_index_dict

# plot the scorer vs feature no
def plot_scorer_vs_feature_no(scorer_train, scorer_test, model,folder_name='./figures', classification=True):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    plt.figure(figsize=(10, 5))
    # x axis is the number of features should start from 1

    plt.plot(range(1, len(scorer_train) + 1), scorer_train, label='Train', color='blue', marker='o')
    plt.plot(range(1, len(scorer_test) + 1), scorer_test, label='Test', color='gold', marker='o')
    plt.xlabel('Number of Features')
    plt.ylabel('scorer (MCC)' if classification else 'scorer (R2)')
    plt.title('MCC vs Number of Features' if classification else 'R2 vs Number of Features')
    plt.legend()
    plt.savefig(f'{folder_name}/{model.__class__.__name__}_MCC_vs_feature_no.png' if classification else f'{folder_name}/{model.__class__.__name__}_R2_vs_feature_no.png')
    plt.close()




def normalize_data(xtrain, xtest):
    scaler = StandardScaler()
    scaler.fit(xtrain)
    xtrain_norm = scaler.transform(xtrain)
    xtest_norm = scaler.transform(xtest)
    xtest_norm = pd.DataFrame(xtest_norm,columns=xtest.columns)
    xtrain_norm = pd.DataFrame(xtrain_norm,columns=xtrain.columns)
    return xtrain_norm, xtest_norm

def remove_constant_features(xtrain, xtest):
    xtrain_constant = xtrain.loc[:, xtrain.nunique() == 1]
    xtrain = xtrain.drop(columns=xtrain_constant.columns)
    xtest = xtest[xtrain.columns]
    return xtrain, xtest

def remove_highly_correlated_features(xtrain, xtest, threshold=0.95):
    corr_matrix = xtrain.corr()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    xtrain = xtrain.drop(columns=to_drop)
    xtest = xtest[xtrain.columns]
    return xtrain, xtest

def remove_low_variance_features(xtrain, xtest, threshold=0.01):
    xtrain_variance = xtrain.var()
    xtest_variance = xtest.var()
    xtrain = xtrain.loc[:, xtrain_variance > threshold]
    xtest = xtest[xtrain.columns]
    return xtrain, xtest




# exahstive feature selection function
def e_feature_selection(model, xtrain, xtest, ytrain, feature_no, verbose=1):
    efs1 = EFS(model,
            min_features=1,
            max_features=feature_no,
            scoring=make_scorer(matthews_corrcoef),
            print_progress=True,
            cv=5,
            n_jobs=-1)
    efs1 = efs1.fit(xtrain, ytrain.values.ravel())
    print('Selected features:', efs1.best_idx_)
    print('feature names: ', efs1.best_feature_names_)
    # print('K_feature:' , efs1.k_feature_idx_)
    Xtrain = xtrain[list(efs1.best_feature_names_)]
    Xtest = xtest[list(efs1.best_feature_names_)]
    return Xtrain, Xtest

def find_best_num_cluster(data, initial_umap_clusters=5, min_clusters=2, max_clusters=10):
    '''
    This function is to find the best number of clusters for the umap clustering
    pass the full set of data_x (train + test) to the function. **not data_y**
    '''
    # Create a UMAP reducer
    reducer = umap.UMAP(n_neighbors=initial_umap_clusters, n_components=2, random_state=42)

    # Apply UMAP to the data
    embedding = reducer.fit_transform(data)

    # Create a DataFrame for the embedding
    umap_df = pd.DataFrame(embedding, columns=['UMAP_1', 'UMAP_2'])

    # store the results in a dictionary
    results_dict = {'n_clusters': [], 'silhouette_score': [], 'davies_bouldin_score': []}
    for n_clusters in range(min_clusters, max_clusters):
        kmeans = KMeans(n_clusters=n_clusters, random_state=60)
        umap_df['Cluster'] = kmeans.fit_predict(embedding)
        silhouette_avg = silhouette_score(embedding, kmeans.labels_)
        db_index = davies_bouldin_score(embedding, kmeans.labels_)
        results_dict['n_clusters'].append(n_clusters)
        results_dict['silhouette_score'].append(silhouette_avg)
        results_dict['davies_bouldin_score'].append(db_index)

        
    # return the row with the maximum difference
    cluster_results = pd.DataFrame(results_dict)
    cluster_results['difference'] = cluster_results['silhouette_score'] - cluster_results['davies_bouldin_score']
    # show the row with the maximum difference
    cluster_results.sort_values(by='difference', ascending=False, inplace=True)
    best_no_cluster = cluster_results.head(1).n_clusters.values[0]
    print(f"best number of clusters: {best_no_cluster}")
    return best_no_cluster, cluster_results

def umap_clustering(data) -> pd.DataFrame:
    # find the best number of clusters
    best_no_cluster, cluster_results = find_best_num_cluster(data)
    # now dow the umap clustering with the best number of clusters
    reducer = umap.UMAP(n_neighbors=best_no_cluster, n_components=2, random_state=42)
    embedding = reducer.fit_transform(data)
    umap_df = pd.DataFrame(embedding, columns=['UMAP_1', 'UMAP_2'])
    # use Kmins to cluster the umap embedding
    kmeans = KMeans(n_clusters=best_no_cluster, random_state=60)
    umap_df['Cluster'] = kmeans.fit_predict(embedding)
    # umap_df['value'] = data_y
    return umap_df


def cluster_split(umap_df, data_x, data_y, test_size=0.2, plot=True, verbose=False):

    '''
    This function is to split the data into train and test for each cluster
    '''
    train_list = []
    test_list = []

    # Loop over each cluster
    for cluster_id, cluster_data in umap_df.groupby('Cluster'):
        if verbose:
            print(f"cluster {cluster_id} has {len(cluster_data)} samples")

        train_part, test_part = train_test_split(
            cluster_data,
            test_size=test_size,          # 20% test split
            random_state=42,        # reproducibility
            shuffle=True
        )
        if len(cluster_data) <= 5:
            print(f"cluster {cluster_id} has less than 5 samples, so it is not used for splitting")
            train_part = cluster_data
            # test_part = pd.DataFrame()
            train_list.append(train_part)
        else:
            train_list.append(train_part)
            # print(f"train part has {len(train_part)} samples")
            test_list.append(test_part)
            # print(f"test part has {len(test_part)} samples")

    # Concatenate all cluster splits
    umap_df_train = pd.concat(train_list)
    umap_df_test = pd.concat(test_list)
    # get the index of the original data_x
    train_df_x = data_x.iloc[umap_df_train.index]
    test_df_x = data_x.iloc[umap_df_test.index]
    train_df_y = data_y.iloc[umap_df_train.index]
    test_df_y = data_y.iloc[umap_df_test.index]
    # show only 3 digits after the decimal point
    print(f"train test ratio: {round(len(test_df_x) / (len(train_df_x) + len(test_df_x)) * 100, 3)}% for test set")
    if plot:
        # plot the umap clustering with respect to train and test
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x='UMAP_1', y='UMAP_2', data=umap_df_train, color='blue', alpha=0.7, s=50)
        sns.scatterplot(x='UMAP_1', y='UMAP_2', data=umap_df_test, color='red', alpha=0.7, s=50)
        plt.title("UMAP Clustering with KMeans", fontsize=16)
        plt.xlabel("UMAP Dimension 1")
        plt.ylabel("UMAP Dimension 2")
        plt.legend(title='Legend', labels=['Train', 'Test'])
        plt.savefig(f'umap_clustering_train_test.png')
        plt.show()
    return train_df_x, test_df_x, train_df_y, test_df_y

def plot_scatter_plot(train_pred, test_pred, train_true, test_true, save_name=None):
    plt.figure(figsize=(10, 8))
    plt.scatter(train_pred, train_true, color='blue', alpha=0.7, s=50)
    plt.scatter(test_pred, test_true, color='gold', alpha=0.7, s=50)
    
    min_line = min(min(train_pred), min(test_pred)) * 1.05
    max_line = max(max(train_pred), max(test_pred)) * 1.05
    # add a diagonal line from or
    plt.plot([min_line, max_line], [min_line, max_line], color='black', linewidth=2)
    plt.title("Scatter Plot of Predicted vs True")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.legend(title='Legend', labels=['Train', 'Test'])
    if save_name:
        plt.savefig(f'scatter_plot_{save_name}.png')
    plt.show()
    return