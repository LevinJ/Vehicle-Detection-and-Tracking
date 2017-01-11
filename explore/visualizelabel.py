import pandas as pd
import matplotlib.pyplot as plt
from utility.vis_utils import vis_grid_withlabels
from sklearn.utils import resample
import matplotlib.image as mpimg
import numpy as np




class VisualizeLabels():
    def __init__(self):

        return
    
   
    def show_label_dist(self):
        df = pd.read_csv('../data/label.csv')
        groups = df.groupby(['label'])
        print(groups['FileName'].count())
#         for name, group in groups:
#             print(name)
#             print(group)
        groups['FileName'].count().plot(kind='bar')
        return
    def show_img_labels(self):
        
        df = pd.read_csv('../data/label.csv')
        img_files,labels = resample(df['FileName'], df['label'], n_samples=4)
        imgs = []
        for img_file in img_files:
            print(img_file)
            imgs.append(mpimg.imread(img_file))
        vis_grid_withlabels(np.asarray(imgs), labels.values)    
        return
        
    def run(self):
        self.show_label_dist()
#         self.show_img_labels()

        
        plt.show()
        return
    


if __name__ == "__main__":   
    obj= VisualizeLabels()
    obj.run()