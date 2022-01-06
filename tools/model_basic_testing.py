from typing import Protocol
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import  cross_val_score
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


class ScikitModel(Protocol):
    def fit(self, X, y, sample_weight=None): ...
    def predict(self, X): ...
    def score(self, X, y, sample_weight=None): ...


def center(string:str, obj_length=84, fill_with=" ")-> str:
    blank = obj_length - len(string)
    filler =""

    if blank>0 :
        for _ in range(int(blank/2)) : filler+=fill_with
        string = "|" + filler + string + filler + (len("|" + filler + string + filler)<=obj_length) *fill_with+"|"

    return string

@ignore_warnings(category=ConvergenceWarning)
def model_basic_testing(model : ScikitModel, X_train : pd.DataFrame, y_train:pd.DataFrame, X_test : pd.DataFrame, y_test:pd.DataFrame, cv_fold=10 )-> None:


    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print(center("WHAT MY ESTIMATOR IS WORTH"))
    print(center("",fill_with="="))
    print(center(" Target Information ", fill_with="-"))
    print(center(""))
    print(center("Prop. of active value in the target : In sample -> " + str( y_train[y_train==1].shape[0]/y_train.shape[0])[:4] + "   Out of sample -> " + str( y_test[y_test==1].shape[0]/y_test.shape[0])[:4]) )
    print(center(""))
    print(center("", fill_with="="))
    print(center(" Scores ",fill_with="-"))
    print(center(""))

    print(center("Accuracy : In sample -> " + str(model.score(X_train, y_train))[:4]+ "  Out of sample -> " + str(model.score(X_test, y_test))[:4],84))

    cv_scores = cross_val_score(model,
                            X_train,
                            y_train,
                            cv=cv_fold)

    print(center("Accuracy (CV) : mean -> " + str(cv_scores.mean())[:4] + "  std.error -> " + str(cv_scores.std())[:4],84))
    print(center("Precision : In sample -> " + str(precision_score(y_train, y_train_pred))[:4]+ "  Out of sample -> " + str(precision_score(y_test, y_test_pred))[:4],84))
    print(center("Recall : In sample -> " + str(recall_score(y_train, y_train_pred))[:4]+ "  Out of sample -> " + str(recall_score(y_test, y_test_pred))[:4],84))
    print(center("F1-Score : In sample -> " + str(f1_score(y_train, y_train_pred))[:4]+ "  Out of sample -> " + str(f1_score(y_test, y_test_pred))[:4],84))
    print(center(""))
    print(center("", fill_with="="))
    print(center(" Classification ",fill_with='-'))
    print(center(""))

    print(center("In sample Confusion Matrix"))
    cm = confusion_matrix(y_train, y_train_pred)
    print(center(str(cm[0,0]) + " values were true negative, (" + str(cm[0,0]/y_train.shape[0])[2:4]+ " %)" ))
    print(center(str(cm[0,1]) + " values were false positive, (" + str(cm[0,1]/y_train.shape[0])[2:4]+ " %)" ))
    print(center(str(cm[1,0]) + " values were false negative, (" + str(cm[1,0]/y_train.shape[0])[2:4]+ " %)" ))
    print(center(str(cm[1,1]) + " values were true positive, (" + str(cm[1,1]/y_train.shape[0])[2:4]+ " %)" ))

    print(center(""))
    print(center("Out of sample Confusion Matrix"))
    cm = confusion_matrix(y_test, y_test_pred)
    print(center(str(cm[0, 0]) + " values were true negative, (" + str(cm[0, 0] / y_test.shape[0])[2:4] + " %)"))
    print(center(str(cm[0, 1]) + " values were false positive, (" + str(cm[0, 1] / y_test.shape[0])[2:4] + " %)"))
    print(center(str(cm[1, 0]) + " values were false negative, (" + str(cm[1, 0] / y_test.shape[0])[2:4] + " %)"))
    print(center(str(cm[1, 1]) + " values were true positive, (" + str(cm[1, 1] / y_test.shape[0])[2:4] + " %)"))

    print(center(""))
    print(center("", fill_with="="))

    return None


















