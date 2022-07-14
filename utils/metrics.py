import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def RSE(pred, true):
    return np.sqrt(np.sum((true-pred)**2)) / np.sqrt(np.sum((true-true.mean())**2))

def CORR(pred, true):
    u = ((true-true.mean(0))*(pred-pred.mean(0))).sum(0) 
    d = np.sqrt(((true-true.mean(0))**2*(pred-pred.mean(0))**2).sum(0))
    return (u/d).mean(-1)

def MAE(pred, true):
    return np.mean(np.abs(pred-true))

def MSE(pred, true):
    return np.mean((pred-true)**2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    
    return mae,mse,rmse,mape,mspe

def get_confusion_matrix(trues, preds):
  labels = range(62)
  conf_matrix = confusion_matrix(trues, preds, labels)
  return conf_matrix
  
def plot_confusion_matrix(conf_matrix):
  plt.imshow(conf_matrix, cmap=plt.cm.Greens)
  indices = range(conf_matrix.shape[0])
  labels = range(62) #[0,1,2,3,4,5,6,7,8,9]
  plt.xticks(indices, labels)
  plt.yticks(indices, labels)
  plt.colorbar()
  plt.xlabel('y_pred')
  plt.ylabel('y_true')
  # 显示数据
  for first_index in range(conf_matrix.shape[0]):
    for second_index in range(conf_matrix.shape[1]):
      plt.text(first_index, second_index, conf_matrix[first_index, second_index])
  plt.savefig('heatmap_confusion_matrix.jpg')
  plt.show()

def get_acc_p_r_f1(trues, preds):
  labels = range(62) #[0,1,2,3,4,5,6,7,8,9]
  TP,FP,FN,TN = 0,0,0,0
  for label in labels:
    preds_tmp = np.array([1 if pred == label else 0 for pred in preds])
    trues_tmp = np.array([1 if true == label else 0 for true in trues])
    # print(preds_tmp, trues_tmp)
    # print()
    # TP预测为1真实为1
    # TN预测为0真实为0
    # FN预测为0真实为1
    # FP预测为1真实为0
    TP += ((preds_tmp == 1) & (trues_tmp == 1)).sum()
    TN += ((preds_tmp == 0) & (trues_tmp == 0)).sum()
    FN += ((preds_tmp == 0) & (trues_tmp == 1)).sum()
    FP += ((preds_tmp == 1) & (trues_tmp == 0)).sum()
  # print(TP, FP, FN)
  precision = TP / (TP + FP)
  recall = TP / (TP + FN)
  f1 = 2 * precision * recall / (precision + recall)
  return precision, recall, f1

def get_acc(trues, preds):
  accuracy = (np.array(trues) == np.array(preds)).sum() / len(trues)
  return accuracy

#print(classification_report(test_trues, test_preds))