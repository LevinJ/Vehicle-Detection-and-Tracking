from preprocess.preparedata import PrepareData
from sklearn import metrics

from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from utility.dumpload import DumpLoad



class SVMModel(PrepareData):

    def __init__(self):
        PrepareData.__init__(self)
        self.do_cross_val = False
        return
    def setClf(self):
        estimator = SVC( kernel='linear', C=10)
        min_max_scaler = preprocessing.MinMaxScaler()
        self.estimator = Pipeline([('scaler', min_max_scaler), ('estimator', estimator)])
        return

  
    def run_croos_validation(self):
        self.setClf()
        features,labels,cv = self.get_cv_folds()
        scores = cross_validation.cross_val_score(self.estimator, features, labels, cv=cv, scoring=self.cv_scoring_mse)
        
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
        if not self.do_cross_val:
            self.run_train_validation()
            return
        self.run_croos_validation()
        
        


        return
    


if __name__ == "__main__":   
    obj= SVMModel()
    obj.run()