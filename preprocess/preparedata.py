import sys
import os
sys.path.insert(0, os.path.abspath('..')) 



from preprocess.preprocessdata import PreprocessData
from sklearn.model_selection import StratifiedShuffleSplit   
from sklearn.model_selection import train_test_split 
import numpy as np


class PrepareData(PreprocessData):    

    def __init__(self):
        PreprocessData.__init__(self)
        return
    def get_cv_folds(self):
        features,labels = self.extract_features_labels()
        ss = StratifiedShuffleSplit(n_splits=10, test_size=0.15)
        return features, labels,ss.split(features, labels)
    
    def get_one_fold(self):
        features,labels = self.extract_features_labels()

        X_train, X_val, y_train, y_val = train_test_split(
            features, labels, test_size=0.15, stratify=labels)
        
        return X_train,y_train,X_val,y_val
    
    def run(self):
        self.get_one_fold()
        


        return
    


if __name__ == "__main__":   
    obj= PrepareData()
    obj.run()