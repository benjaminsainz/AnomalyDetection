import numpy as np
import math
import random
import pandas as pd
import glob
from sklearn.metrics import pairwise_distances, roc_auc_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils.validation import check_X_y, check_array
from sklearn.preprocessing import OneHotEncoder
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from brminer import BRM

class BRM_Modified(BRM):
    def __init__(self, dissimilarity_measure = 'euclidean', normalization_type = None):
        BRM.__init__(self, classifier_count=100, bootstrap_sample_percent=100, use_bootstrap_sample_count=False, bootstrap_sample_count=0, use_past_even_queue=False, max_event_count=3, alpha=0.5, user_threshold=95)
        self.dissimilarity_measure = dissimilarity_measure
        self._normalization_type = normalization_type

    def score_samples(self, X):
        X_test = np.array(X)
        result = []
        batch_size = 100
        for i in range(min(len(X_test), batch_size), len(X_test) + batch_size, batch_size):
            current_X_test = X_test[[j for j in range(max(0, i-batch_size), min(i, len(X_test)))]]
            current_similarity = np.average([np.exp(-np.power(np.amin(pairwise_distances(current_X_test, self._centers[i], metric=self.dissimilarity_measure), axis=1)/self._max_dissimilarity, 2)/(self._sd[i])) for i in range(len(self._centers))], axis=0)
            result = result + [j for j in list(map(self._evaluate, current_similarity))]
        return result

    def pretraining_data_processing(self, X, y):
        if y is not None:
            X_train, _ = check_X_y(X, y)
        else:
             X_train = check_array(X)
        self._similarity_sum = 0
        self._is_threshold_Computed = False
        self.n_features_in_ = X_train.shape[1]
        X_train = pd.DataFrame(X_train)
        return X_train

    def fit(self, X, y = None):
        X_train = self.pretraining_data_processing(X, y)
        self._max_dissimilarity = math.sqrt(self.n_features_in_)
        self._sd = np.empty(0)
        sampleSize = int(self.bootstrap_sample_count) if (self.use_bootstrap_sample_count) else int(0.01 * self.bootstrap_sample_percent * len(X_train));
        self._centers = np.empty((0, sampleSize, self.n_features_in_))
        list_instances = X_train.values.tolist()
        for i in range(0, self.classifier_count):            
            centers = random.choices(list_instances, k=sampleSize)
            self._centers = np.insert(self._centers, i, centers, axis=0)
            self._sd = np.insert(self._sd, i, 2*(np.mean(pairwise_distances(centers, centers, metric=self.dissimilarity_measure))/self._max_dissimilarity)**2)
        return self

def normalization(X_train_raw, X_test_raw, normal_type):
    if normal_type == 'no':
        return X_train_raw, X_test_raw
    elif normal_type == 'minmax':
        X_train = MinMaxScaler().fit_transform(X_train_raw)
        X_test = MinMaxScaler().fit_transform(X_test_raw)
    elif normal_type == 'standard':
        X_train = StandardScaler().fit_transform(X_train_raw)
        X_test = StandardScaler().fit_transform(X_test_raw)
    return X_train, X_test

def rebuild_preprocessed_train_test(codAllDataAgain, objInTrain, train, test, normal_type):
    X_train = codAllDataAgain[:objInTrain]
    y_train = train.values[:, -1]
    X_test = codAllDataAgain[objInTrain:]
    y_test = test.values[:, -1]
    X_train, X_test = normalization(X_train, X_test, normal_type)
    return X_train, X_test, y_train, y_test

def preprocess_dataset(train, test, normal_type): 
    ohe = OneHotEncoder(sparse=True)
    allData = pd.concat([train, test], ignore_index=True, sort =False, axis=0)
    AllDataWihoutClass = allData.iloc[:, :-1]
    AllDataWihoutClassOnlyNominals = AllDataWihoutClass.select_dtypes(include=['object'])
    AllDataWihoutClassNoNominals = AllDataWihoutClass.select_dtypes(exclude=['object'])
    encAllDataWihoutClassNominals = ohe.fit_transform(AllDataWihoutClassOnlyNominals)
    encAllDataWihoutClassNominalsToPanda = pd.DataFrame(encAllDataWihoutClassNominals.toarray())
    if AllDataWihoutClassOnlyNominals.shape[1] > 0:
        codAllDataAgain = pd.concat([encAllDataWihoutClassNominalsToPanda, AllDataWihoutClassNoNominals], ignore_index=True, sort =False, axis=1)
    else:
        codAllDataAgain = AllDataWihoutClass
    X_train, X_test, y_train, y_test = rebuild_preprocessed_train_test(codAllDataAgain, len(train), train, test, normal_type)
    return X_train, X_test, y_train, y_test

def raw_train_test_data(folder):
    trainFile = glob.glob('{}/*tra.csv'.format(folder))[0]
    testFile = glob.glob('{}/*tst.csv'.format(folder))[0]
    raw_train = pd.read_csv(trainFile, header = None) 
    raw_test = pd.read_csv(testFile, header = None) 
    return raw_train, raw_test

def run_classification_model_brm(X_train, y_train, X_test, y_test, metric_list, model):
    model.fit(X_train, y_train)
    pred = model.score_samples(X_test)
    auc = roc_auc_score(y_test,  pred)
    metric_list.append(auc if auc > .5 else 1 - auc)
    return metric_list

def run_classification_model_competitor(X_train, y_train, X_test, y_test, metric_list, model):
    trained_model = model.fit(X_train)
    pred = trained_model.predict(X_test)
    auc = roc_auc_score(y_test,  pred)
    metric_list.append(auc if auc > .5 else 1 - auc)
    return metric_list

def run_gmm(X_train, y_train, X_test, y_test, metrics_list, model):
    trained_model = model.fit(X_train)
    pred = trained_model.score_samples(X_test)
    auc = roc_auc_score(y_test,  pred)
    metrics_list.append(auc if auc > .5 else 1 - auc)
    return metrics_list

def model_objects(normalization_type):
    original_model = BRM_Modified(dissimilarity_measure = 'euclidean', normalization_type = normalization_type)
    correlation_model = BRM_Modified(dissimilarity_measure = 'correlation', normalization_type = normalization_type)
    cosine_model = BRM_Modified(dissimilarity_measure = 'cosine', normalization_type = normalization_type)
    manhattan_model = BRM_Modified(dissimilarity_measure = 'manhattan', normalization_type = normalization_type)
    gmm_model = GaussianMixture()
    isof_model = IsolationForest()
    ocsvm_model = OneClassSVM()
    models = [original_model, correlation_model, cosine_model, manhattan_model, gmm_model, isof_model, ocsvm_model]
    return models

def run_all_clf(X_train, y_train, X_test, y_test, clf_metrics, normalization_type):
    models = model_objects(normalization_type)
    brm_auc_all_original = run_classification_model_brm(X_train, y_train, X_test, y_test, clf_metrics[0], models[0])
    brm_auc_all_correlation = run_classification_model_brm(X_train, y_train, X_test, y_test, clf_metrics[1], models[1])
    brm_auc_all_cosine = run_classification_model_brm(X_train, y_train, X_test, y_test, clf_metrics[2], models[2])
    brm_auc_all_manhattan = run_classification_model_brm(X_train, y_train, X_test, y_test, clf_metrics[3], models[3])
    gmm_auc_all = run_gmm(X_train, y_train, X_test, y_test, clf_metrics[4], models[4])
    iso_auc_all = run_classification_model_competitor(X_train, y_train, X_test, y_test, clf_metrics[5], models[5])
    osv_auc_all = run_classification_model_competitor(X_train, y_train, X_test, y_test, clf_metrics[6], models[6])
    return [brm_auc_all_original, brm_auc_all_correlation, brm_auc_all_cosine, brm_auc_all_manhattan, gmm_auc_all, iso_auc_all, osv_auc_all]

def export_results(folders, auc_all, normal_type):
    results = pd.DataFrame()
    results['dataset'] = [filename.split('/')[1] for filename in folders]
    results['brm_original'] = auc_all[0]
    results['brm_correlation'] = auc_all[1]
    results['brm_cosine'] = auc_all[2]
    results['brm_manhattan'] = auc_all[3]
    results['gmm'] = auc_all[4]
    results['isof'] = auc_all[5]
    results['ocsvm'] = auc_all[6]
    results.to_csv('benchmark_{}_normalization.csv'.format(normal_type), index=False)

def normalization_clf_test(folders, normal_type, auc_all):
    for folder in folders:
        print('Processing {} dataset with {} normalization.'.format(folder.split('/')[1], normal_type))
        raw_train, raw_test = raw_train_test_data(folder)
        X_train, X_test, y_train, y_test = preprocess_dataset(raw_train, raw_test, normal_type)
        auc_all = run_all_clf(X_train, y_train, X_test, y_test, auc_all, normal_type)
    return auc_all

def run_benchmark():
  folders = sorted(glob.glob('Datasets/*'))
  for normal_type in ['no', 'minmax', 'standard']:
    auc_all = [[], [], [], [], [], [], []]
    auc_all = normalization_clf_test(folders, normal_type, auc_all)
    export_results(folders, auc_all, normal_type)

if __name__ == "__main__":
    run_benchmark()
