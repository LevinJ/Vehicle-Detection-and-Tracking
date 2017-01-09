import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from preprocess.spatial_bin import SpatialBin
from preprocess.color_histogram import ColorHistogram
import pandas as pd
from preprocess.hogfeature import HOGFeature
from utility.dumpload import DumpLoad



class PreprocessData(SpatialBin, ColorHistogram, HOGFeature):
    def __init__(self):
        SpatialBin.__init__(self)
        ColorHistogram.__init__(self)
        HOGFeature.__init__(self)
#         self.used_features = ['hog', 'color', 'raw']
        self.used_features = ['hog']
        self.label = 'label'
        return
    
    def __extract_hog_features(self, img_files):
        dump_load = DumpLoad('../data/hog_features.pickle')
        if dump_load.isExisiting():
            return dump_load.load()
        features = []
        for fname in img_files:
            img = mpimg.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            feature = self.get_hog_features(gray)
            features.append(feature)
        features = np.asanyarray(features)
        dump_load.dump(features)
        return features
    def __extract_color_hist_features(self, img_files):
        dump_load = DumpLoad('../data/color_hist_features.pickle')
        if dump_load.isExisiting():
            return dump_load.load()
        features = []
        for fname in img_files:
            img = mpimg.imread(fname)
            _, _, _, _, hist_features = self.color_hist(img, nbins=32, bins_range=(0, 256))
            features.append(hist_features)
        features = np.asanyarray(features)
        dump_load.dump(features)
        return features
    def __extract_bin_spatialfeatures(self, img_files):
        dump_load = DumpLoad('../data/bin_spatial_features.pickle')
        if dump_load.isExisiting():
            return dump_load.load()
        features = []
        for fname in img_files:
            img = mpimg.imread(fname)
            bs_features = self.bin_spatial(img, color_space='RGB', size=(32, 32))
            features.append(bs_features)
        features = np.asanyarray(features)
        dump_load.dump(features)
        return features
       
    def extract_features_labels(self):
        df = pd.read_csv('../data/label.csv', index_col=0)
        img_files = df['FileName']
        hog_features = self.__extract_hog_features(img_files)
        color_hist_features = self.__extract_color_hist_features(img_files)
        bs_features = self.__extract_bin_spatialfeatures(img_files)
        feature_list = []
        if 'hog' in self.used_features:
            feature_list.append(hog_features)
            
        if 'color' in self.used_features:
            feature_list.append(color_hist_features)
        if 'raw' in self.used_features:
            feature_list.append(bs_features)
      
        features = np.concatenate(feature_list, axis=1)
        
        labels = df['label'].values
            
        return features,labels
   
    def run(self):
        
        res = self.extract_features_labels()
       
       
       
        plt.show()

        return



if __name__ == "__main__":   
    obj= PreprocessData()
    obj.run()