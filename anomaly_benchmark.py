import glob
import brminer
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


def rebuild_preprocessed_train_test(codAllDataAgain, objInTrain, train, test, normal_type):
  X_train = codAllDataAgain[:objInTrain]
  y_train = train.values[:, -1]
  X_test = codAllDataAgain[objInTrain:]
  y_test = test.values[:, -1]
  X_train, X_test = normalization(X_train, X_test, normal_type)
  return X_train, X_test, y_train, y_test

def normalization(X_train_raw, X_test_raw, normal_type):
  if normal_type == 'no':
    X_train = X_train_raw
    X_test = X_test_raw
  elif normal_type == 'minmax':
    X_train = MinMaxScaler().fit_transform(X_train_raw)
    X_test = MinMaxScaler().fit_transform(X_test_raw)
  elif normal_type == 'standard':
    X_train = StandardScaler().fit_transform(X_train_raw)
    X_test = StandardScaler().fit_transform(X_test_raw)
  return X_train, X_test

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

def run_brm(X_train, y_train, X_test, y_test, brm_auc_all):
  brm_model = brminer.BRM().fit(X_train, y_train)
  brm_pred = brm_model.score_samples(X_test)
  brm_auc = roc_auc_score(y_test,  brm_pred)
  brm_auc_all.append(brm_auc if brm_auc > .5 else 1 - brm_auc)
  return brm_auc_all

def run_gmm(X_train, y_train, X_test, y_test, gmm_auc_all):
  gmm_model = GaussianMixture().fit(X_train)
  gmm_pred = gmm_model.score_samples(X_test)
  gmm_auc = roc_auc_score(y_test,  gmm_pred)
  gmm_auc_all.append(gmm_auc if gmm_auc > .5 else 1 - gmm_auc)
  return gmm_auc_all

def run_iso(X_train, y_train, X_test, y_test, iso_auc_all):
  iso_model = IsolationForest().fit(X_train)
  iso_pred = iso_model.predict(X_test)
  iso_auc = roc_auc_score(y_test,  iso_pred)
  iso_auc_all.append(iso_auc if iso_auc > .5 else 1 - iso_auc)
  return iso_auc_all

def run_osv(X_train, y_train, X_test, y_test, osv_auc_all):
  osv_model = OneClassSVM().fit(X_train)
  osv_pred = osv_model.predict(X_test)
  osv_auc = roc_auc_score(y_test,  osv_pred)
  osv_auc_all.append(osv_auc if osv_auc > .5 else 1 - osv_auc)
  return osv_auc_all

def run_all_clf(X_train, y_train, X_test, y_test, clf_metrics):
  brm_auc_all, gmm_auc_all, iso_auc_all, osv_auc_all = clf_metrics
  brm_auc_all = run_brm(X_train, y_train, X_test, y_test, brm_auc_all)
  gmm_auc_all = run_gmm(X_train, y_train, X_test, y_test, gmm_auc_all)
  iso_auc_all = run_iso(X_train, y_train, X_test, y_test, iso_auc_all)
  osv_auc_all = run_osv(X_train, y_train, X_test, y_test, osv_auc_all)
  return brm_auc_all, gmm_auc_all, iso_auc_all, osv_auc_all

def export_results(folders, brm_auc_all, gmm_auc_all, iso_auc_all, osv_auc_all, normal_type):
  results = pd.DataFrame()
  results['dataset'] = [filename.split('/')[1] for filename in folders]
  results['brm'] = brm_auc_all
  results['gmm'] = gmm_auc_all
  results['isof'] = iso_auc_all
  results['ocsvm'] = osv_auc_all
  results.to_csv('benchmark_{}_normalization.csv'.format(normal_type), index=False)

def normalization_clf_test(folders, normal_type, clf_metrics):
  brm_auc_all, gmm_auc_all, iso_auc_all, osv_auc_all = clf_metrics
  for folder in folders:
    print('Processing {} dataset with {} normalization.'.format(folder.split('/')[1], normal_type))
    raw_train, raw_test = raw_train_test_data(folder)
    X_train, X_test, y_train, y_test = preprocess_dataset(raw_train, raw_test, normal_type)
    clf_metrics = [brm_auc_all, gmm_auc_all, iso_auc_all, osv_auc_all]
    brm_auc_all, gmm_auc_all, iso_auc_all, osv_auc_all = run_all_clf(X_train, y_train, X_test, y_test, clf_metrics)
  return brm_auc_all, gmm_auc_all, iso_auc_all, osv_auc_all

def run_benchmark():
  folders = sorted(glob.glob('Datasets/*'))
  for normal_type in ['no', 'minmax', 'standard']:
    clf_metrics = [[], [], [], []]
    brm_auc_all, gmm_auc_all, iso_auc_all, osv_auc_all = normalization_clf_test(folders, normal_type, clf_metrics)
    export_results(folders, brm_auc_all, gmm_auc_all, iso_auc_all, osv_auc_all, normal_type)

if __name__ == "__main__":
  run_benchmark()
