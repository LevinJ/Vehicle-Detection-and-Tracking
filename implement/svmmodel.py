from preprocess.preparedata import PrepareData
from sklearn import metrics

from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from utility.dumpload import DumpLoad
from sklearn.model_selection import GridSearchCV
import numpy as np



class SVMModel(PrepareData):

    def __init__(self):
        PrepareData.__init__(self)
        self.do_grid_search = True
        self.do_cross_val = True
        
        return
    def setClf(self):
        estimator = SVC( kernel='linear', C=10)
        min_max_scaler = preprocessing.MinMaxScaler()
        self.estimator = Pipeline([('scaler', min_max_scaler), ('estimator', estimator)])
        return
    def run_grid_search(self):
        self.setClf()
        features,labels,cv = self.get_cv_folds()
        parameters = {'estimator__C':[0.01,0.1,0.5, 1, 10, 100,1000,10000]}
        estimator = GridSearchCV(self.estimator, parameters, cv=cv,n_jobs=8, scoring='f1', verbose = 500)
        estimator.fit(features, labels)
        print('estimaator parameters: {}'.format(estimator.get_params))
        print("cv reuslt{}".format(estimator.cv_results_ ))
        print('Best parameters: {}'.format(estimator.best_params_))
        print('Best Scores: {}'.format(estimator.best_score_))
#         print('Score grid: {}'.format(estimator.grid_scores_ ))
#         for i in estimator.grid_scores_ :
#             print('parameters: {}'.format(i.parameters ))
#             print('mean_validation_score: {}'.format(np.absolute(i.mean_validation_score)))
#             print('cv_validation_scores: {}'.format(np.absolute(i.cv_validation_scores) ))
#         return

  
    def run_croos_validation(self):
        self.setClf()
        features,labels,cv = self.get_cv_folds()
        scores = cross_validation.cross_val_score(self.estimator, features, labels, cv=cv, scoring='f1',n_jobs=4,
                    verbose=100)
        
        #use recall as evaluation metrics
#         scores = cross_validation.cross_val_score(self.estimator, features, labels, cv=cv, scoring='f1')
        print("cross validation scores: means, {}, std, {}, details,{}".format(scores.mean(), scores.std(), scores))
        return  scores.mean()
    def predict_sliding_window(self, img):
        dump_load = DumpLoad('../data/smvmodel.pickle')
        if  not hasattr(self, 'estimator'):
            if dump_load.isExisiting():
                self.estimator = dump_load.load()
            else:
                self.run_train_validation() 
                dump_load.dump(self.estimator)

        features = self.extract_features(img)
        res = self.estimator.predict(features.reshape(1,-1))
        return res[0]
    def run_train_validation(self):
        self.setClf()
        X_train,y_train,X_val,y_val = self.get_one_fold()
        self.estimator.fit(X_train,y_train)
        
        y_train_pred = self.estimator.predict(X_train)
        y_val_pred = self.estimator.predict(X_val)
        
        # Precision
        train_precision = metrics.precision_score(y_train, y_train_pred)
        validation_precision = metrics.precision_score(y_val, y_val_pred)
        print("train_precision: {}, validation_precision: {}".format(train_precision, validation_precision))
        
        # Recall
        
        train_recall = metrics.recall_score(y_train, y_train_pred)
        validation_recall = metrics.recall_score(y_val, y_val_pred)
        print("train_recall: {}, validation_recall: {}".format(train_recall, validation_recall))
        
        # f1
        train_f1 = metrics.f1_score(y_train, y_train_pred)
        validation_f1 = metrics.f1_score(y_val, y_val_pred)
        
        print("train_f1: {}, validation_f1: {}".format(train_f1, validation_f1))
        
        
        return 
    
    

    def run(self):
        if self.do_grid_search:
            self.run_grid_search()
            return
        if self.do_cross_val:
            self.run_croos_validation()
            return
        
        self.run_train_validation()
        
        


        return
    


if __name__ == "__main__":   
    obj= SVMModel()
    obj.run()