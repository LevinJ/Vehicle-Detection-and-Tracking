import sys
import os
sys.path.insert(0, os.path.abspath('..')) 



from preprocess.preprocessdata import PreprocessData
from sklearn.model_selection import StratifiedShuffleSplit    


class PrepareData(PreprocessData):    

    def __init__(self):
        PreprocessData.__init__(self)
        return
    def get_cv_folds(self):
        features,labels = self.extract_features_labels()
        ss = StratifiedShuffleSplit(n_splits=5, test_size=0.15)
        return features, labels,ss.split(features, labels)
    
    def get_one_fold(self):
        features,labels = self.extract_features_labels()
        ss = StratifiedShuffleSplit(n_splits=2, test_size=0.15)
        folds = []
        used_foldid = 0
        for train_index, test_index in ss.split(features, labels):
            folds.append((train_index, test_index))
            break
       
        train_index = folds[used_foldid][0]
        val_index = folds[used_foldid][1]
       
        
        X_train = features[train_index]
        y_train= labels[train_index]
        
        
        X_val = features[val_index]
        y_val = labels[val_index]
        
        return X_train,y_train,X_val,y_val
    
    def run(self):
        self.get_one_fold()
        


        return
    


if __name__ == "__main__":   
    obj= PrepareData()
    obj.run()