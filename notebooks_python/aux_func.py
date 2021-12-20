import pandas as pd
import numpy as np
from sklearn import tree
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix,roc_curve, recall_score,precision_score, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import scikitplot as skplt
import matplotlib.pyplot as plt
from scikitplot.metrics import plot_lift_curve

# Función para imprimir métricas
def print_metrics(y_test, y_pred):
    print("------------------------------------------")
    print("The accuracy score is: ", round(accuracy_score(y_test, y_pred),3))
    print("------------------------------------------")
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    print("The F1 score is: ",round(metrics.auc(fpr, tpr),3))
    print("------------------------------------------")
    print("The recall is: ",round(recall_score(y_test,y_pred),3))
    print("------------------------------------------")
    print("The precision score is: ",round(precision_score(y_test,y_pred),3))
    
# Función para matriz de confusiones 
def matriz_confusion(y_test, y_pred, size_figure = [13,13]):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7,7))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'BuPu');
    plt.ylabel('Realidad');
    cm2=[[cm[0][0]/sum(cm[0]),cm[0][1]/sum(cm[0])],[cm[1][0]/sum(cm[1]),cm[1][1]/sum(cm[1])]]
    plt.figure(figsize=(7,7))
    sns.heatmap(cm2, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'BuPu');
    plt.ylabel('Realidad')

# Función para curva roc 
def curva_roc(y_test,y_pred_proba):
# Mantenemos las probabilidades
    yhat = y_pred_proba[:, 1]

    # calculamos la curva de ROC
    fpr, tpr, thresholds = roc_curve(y_test, yhat)

    # gmeans
    gmeans = np.sqrt(tpr * (1-fpr))
    ix = np.argmax(gmeans)
    print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))

    # ploteamos la curva roc
    plt.plot([0,1], [0,1], linestyle='--', label='No Skill')
    plt.plot(fpr, tpr, marker='.', label='Selected Model')
    plt.scatter(fpr[ix], tpr[ix], s=100, marker='o', color='black', label='Best')

    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()

# Función precision-recall curve 
def curva_pr (y_test, y_pred_proba):
    
    yhat = y_pred_proba[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, yhat)
    fpr, tpr, thresholds = roc_curve(y_test, yhat)
    # f score:
    fscore = (2 * precision * recall) / (precision + recall)
    # máximo:
    ix = np.argmax(fscore)
    #print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))
    # curva roc
    no_skill = len(y_test[y_test== 1]) / len(y_test)
    plt.plot([0,1], [no_skill, no_skill], linestyle='-', label='No Skill')
    plt.plot(recall, precision, marker='*', label='Model')
    plt.scatter(recall[ix], precision[ix], s=100, marker='o', color='black', label='Best')
    # ejes
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.show();

# Función para curva de ganancia
def ganancia(modelo, y_test, y_pred_proba, X_test):
    prob_predictions = modelo.predict_proba(X_test)
    yhat = y_pred_proba[:, 1]
    skplt.metrics.plot_cumulative_gain(y_test, prob_predictions)
    plt.show()
    
# Función para curva lift    
def curva_lift(modelo, y_test, y_pred_proba, X_test):
    prob_predictions = modelo.predict_proba(X_test)
    plot_lift_curve(y_test, prob_predictions)
    plt.show()
    
# Confusiones optimizada:  
def matriz_confusion_optimizada(model, X_test, y_test, y_pred_proba):
    prob_predictions = model.predict_proba(X_test)
    yhat = prob_predictions[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, yhat)
    gmeans = np.sqrt(tpr * (1-fpr))
    ix = np.argmax(gmeans)
    y_pred_best = (y_pred_proba[:,1] >= thresholds[ix]).astype(int)
    # Ploteamos la matriz:
    cm = confusion_matrix(y_test, y_pred_best)
    plt.figure(figsize=(7,7))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'BuPu');
    plt.ylabel('Realidad');
    cm2=[[cm[0][0]/sum(cm[0]),cm[0][1]/sum(cm[0])],[cm[1][0]/sum(cm[1]),cm[1][1]/sum(cm[1])]]
    plt.figure(figsize=(7,7))
    sns.heatmap(cm2, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'BuPu');
    plt.ylabel('Realidad')

    
def print_metrics_optimized(model, X_test, y_test, y_pred_proba):
    # Métricas optimizadas
    prob_predictions = model.predict_proba(X_test)
    yhat = prob_predictions[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, yhat)
    gmeans = np.sqrt(tpr * (1-fpr))
    ix = np.argmax(gmeans)
    y_pred_best = (y_pred_proba[:,1] >= thresholds[ix]).astype(int)
    print("------------------------------------------")
    print("The accuracy score is: ", round(accuracy_score(y_test, y_pred_best),3))
    print("------------------------------------------")
    print("The F1 score is: ",round(metrics.auc(fpr, tpr),3))
    print("------------------------------------------")
    print("The recall is: ",round(recall_score(y_test, y_pred_best),3))
    print("------------------------------------------")
    print("The precision score is: ",round(precision_score(y_test, y_pred_best),3))
