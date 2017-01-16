import sys
import os
sys.path.insert(0, os.path.abspath('..')) 

from utility.dumpload import DumpLoad
import matplotlib.pyplot as plt
from preprocess.preparedata import PrepareData
from sklearn import metrics
from utility.confusionmatix import show_confusion_matrix
from utility.vis_utils import vis_grid
from postprocess.sliding_window import SlidingWindow

class AnalyzePrediction(PrepareData):    

    def __init__(self):
        PrepareData.__init__(self)
        SlidingWindow.__init__(self)
        return
    def __load_model(self):
        dump_load = DumpLoad('../data/smvmodel.pickle')
        self.model = dump_load.load()

        return
    def predict(self):
        X,y = self.extract_features_labels()
        images, img_files = self.extract_images()
        y_pred = self.model.predict(X)
        f1 = metrics.f1_score(y, y_pred)
        

        print("f1 score {}".format(f1))
        
        vehicle = (y==1) &(y_pred == 0) 
        vehicle_img = vis_grid(images[vehicle])
        print('vehicle {}'.format(img_files[vehicle]))
        
        nonvehicle = (y==0) & (y_pred == 1) 
        nonvehicle_img = vis_grid(images[nonvehicle])
        print('non vihiecle {}'.format(img_files[nonvehicle]))
        
        _,(ax1, ax2) = plt.subplots(2,1)
        ax1.imshow(vehicle_img)
        ax1.set_title('vehicle')
        
        ax2.imshow(nonvehicle_img)
        ax2.set_title('nonvehicle')
        
        
        
        
        
        alphabet = ['non-vehicle','vehicle']
        show_confusion_matrix(y, y_pred, alphabet)
        return
    def predict_roi(self):
        return
    
    
    def run(self):
        self.__load_model()
        self.predict()
        plt.show()
        
        
        


        return
    


if __name__ == "__main__":   
    obj= AnalyzePrediction()
    obj.run()