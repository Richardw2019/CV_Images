import pandas as pd
import numpy as np
import json
import os
import shutil 
import matplotlib.pyplot as plt
import copy
import requests
import glob
import cv2
import copy
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve
from zipfile import ZipFile
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn import metrics

## create_partitions (might be a predefined function) [LG] (DONE)
## a function that takes all of the labeled data and partition it into lists
## input: path to labeled data folder, size of k(# of cross validation)
## output: k lists of partitioned data 
def create_partitions(data_path,k):
  #  data = f.read().split('\n')
  data = [i for i in os.listdir(data_path) if '.zip' not in i]
  data = np.array(data)  #convert array to numpy type array
  cv = KFold(n_splits=k, shuffle=True)
  splits = cv.split(data)

  split_filenames = []
  for split_idx, split in enumerate(splits):
    # get actual filenames
    train_filenames = data[split[0]]
    test_filenames = data[split[1]]
    split_filenames.append({"split": split_idx, "train_files": train_filenames, "test_files": test_filenames})
  
  return split_filenames


"""
  This function takes as input a dictionary of CV split partitions, as created
  e.g. by the function create_split_zip(), as well as a split index and a path
  to the data in question. This function then creates a zip archive for the 
  training and test sets, deposits those zips at data_path, and returns
  two strings which are the paths to the resulting zip files.
"""
def create_split_zip(partitions_dict, split_idx, data_path):

  # Paths for the zip files that will be created
  test_zip_path = data_path + 'testzip_' + str(split_idx)+'.zip'
  train_zip_path =  data_path + 'trainzip_' + str(split_idx)+'.zip'

  # Get test_list of files (with full path)
  test_list = [data_path + '/'+ file for file in partitions_dict[split_idx]['test_files']]
  train_list = [data_path + '/'+ file for file in partitions_dict[split_idx]['train_files']]

  for file in test_list:
    # print(file) # for debug
    with ZipFile(test_zip_path, "a") as f:
      # filename is the name of the file without full path
      filename = file[len(data_path):]
      # print('Writing ',file[len(data_path):],'to ',test_zip_path) #for debug
      f.write(file, filename)

  for file in train_list:
    with ZipFile(train_zip_path, "a") as f:
      # filename is the name of the file without full path
      filename = file[len(data_path):]
      # print('Writing ',file[len(data_path):],'to ',train_zip_path) #for debug
      f.write(file, filename)

  return [test_zip_path, train_zip_path]


"""
  A function which creates and trains a classifier on 
  watson visual recognition using the api.
  The two zip file inputs should be full paths to the zip files.
"""
def create_and_train_model (good_training_zip, bad_training_zip, apikey, url):

  visual_recognition.set_service_url(url)

  with open(good_training_zip, 'rb') as good, open(bad_training_zip, 'rb') as bad:
      model = visual_recognition.create_classifier(
          'cv_images',
          positive_examples={'good':good},
          negative_examples=bad).get_result()
  print(json.dumps(model, indent=2))
  return(model["classifier_id"])


#Sends test zips to its specific model and stores the results in a list
def test_model(model_ids, joinedlist):
  scores = []
  j = 0
  for mod in model_ids:
      print("Split:", mod['split'], "Model id:", mod['id'])
      for i in range(2):
        # print(joinedlist[j][i])
        with open(joinedlist[j][i], 'rb') as images_file:
          classes = visual_recognition.classify(
              images_file=images_file,
              threshold='0',
              classifier_ids=mod['id'])
          classes_results = classes.get_result()     
        scores.append(classes_results)
      j = j + 1
  return scores



def get_results(scores):
  final_results = []
  for i in range (0,16):
    results = copy.deepcopy(scores[i])
    del results['custom_classes']
    del results['images_processed']
    for key_1 in results.values():
      for key_2 in key_1:
        if (i % 2) == 0:
          resdict= {'image':key_2['image'], 'actualscore':1,'predictedscore':key_2['classifiers'][0]['classes'][0]['score']}
          final_results.append(resdict)
        if (i % 2) == 1:
          resdict= {'image':key_2['image'], 'actualscore':0,'predictedscore':key_2['classifiers'][0]['classes'][0]['score']}
          final_results.append(resdict)

  df_results = pd.DataFrame(final_results)

  return df_results






def evaluate_results(df_results):
  y_test = df_results.actualscore
  y_prob = df_results.predictedscore

  #PRECISON-RECALL CURVE
  def fmt_rat(flt,dec=3):
      flt = str(round(flt,dec))
      return flt

  # Precision-Recall Analysis:
  precision, recall, eps = metrics.precision_recall_curve(y_test, y_prob,pos_label=1)


  # precision recall curve
  fig, ax = plt.subplots();
  ax.plot(recall, precision, 'b-')
  ax.set(xlabel='recall', ylabel='precision',ylim=(0,None))
  ax.set_aspect('equal')
  plt.title("Precision-Recall Curve")

  # annotations
  idx = np.linspace(0,len(eps)-3,20).astype(int)-1
  ax.plot(recall[idx], precision[idx], 'ko')
  for i in idx:
      ax.annotate(fmt_rat(eps[i],2), (recall[i]+.02, precision[i]+.02), 
                  horizontalalignment='left', 
                  verticalalignment='bottom',
                  fontsize=12
                )
      
  fig.set_size_inches(6,6)
  plt.show()
  plt.clf()


  #F1 score
  f1 = 2 * precision * recall / (precision + recall)
  f1=f1[f1==f1]
  f1max_ind = np.argmax(f1)
  print("Maximum F1 score: " + fmt_rat(f1[f1max_ind]))
  print("Precision at max: " + fmt_rat(precision[f1max_ind]))
  print("Recall at max: " + fmt_rat(recall[f1max_ind]))
  print("Optimal threshold: " + fmt_rat(eps[f1max_ind]))

  #Confusion matrix
  cm =  confusion_matrix(y_test, (y_prob>0.208).astype(int)).transpose()
  print("Confusion matrix for threshold=0.208: \n", cm)


  # ROC-AUC Analysis
  fpr, tpr, thresholds = metrics.roc_curve(df_results.actualscore,df_results.predictedscore, pos_label=1)
  roc_auc_test = metrics.roc_auc_score(df_results.actualscore,df_results.predictedscore)

  # roc curve
  fig, ax = plt.subplots() 
  fig.set_size_inches(6,6)
  ax.plot(fpr, tpr, 'b-', label='Model, AUC: ' + fmt_rat(roc_auc_test))
  ax.plot([0,1],[0,1],'k--', label='Chance, AUC: 0.5')
  ax.set(xlabel='FPR', ylabel='TPR')
  ax.set_aspect('equal')

  # annotations
  idx = np.linspace(0,len(thresholds)-3,15).astype(int)-1
  ax.plot(fpr[idx], tpr[idx], 'ko')
  for i in idx:
      ax.annotate(fmt_rat(thresholds[i],2), (fpr[i]+0.02, tpr[i]-.05), 
                  horizontalalignment='left', 
                  verticalalignment='bottom',
                  fontsize=12
                )

  ax.legend()
  plt.title("ROC Curve")
  plt.show()
  plt.clf()


#saving the dataframe to a csv CHANGE THE NAME
def save_results(df_results):
  df_results.to_csv("Spring20_CrossValidation_Result_Fall2020.csv")
  src = 'Spring20_CrossValidation_Result_Fall2020.csv'
  dst =  '/content/drive/My Drive/Projects/CV Imaging/Data'
  shutil.copy(src, dst)