
import numpy as np
import cv2
import matplotlib.pyplot as plt
from preprocess.spatial_bin import SpatialBin
from preprocess.color_histogram import ColorHistogram
import pandas as pd
from preprocess.hogfeature import HOGFeature
from utility.dumpload import DumpLoad
import os




class PreprocessData(SpatialBin, ColorHistogram, HOGFeature):
    def __init__(self):
        SpatialBin.__init__(self)
        ColorHistogram.__init__(self)
        HOGFeature.__init__(self)
        self.feature_pickle = '../data/features.pickle'
#         self.used_features = ['hog', 'color', 'raw']
        self.used_features = ['hog']
        self.label = 'label'
        return
    
    
    def extract_images(self):
        dump_load = DumpLoad('../data/images.pickle')
        if dump_load.isExisiting():
            return dump_load.load()
        features = []
        df = pd.read_csv('../data/label.csv', index_col=0)
        img_files = df['FileName']
        for fname in img_files:
            img = cv2.imread(fname)
            img = img[...,::-1]
            features.append(img)
        features = np.asanyarray(features)
        dump_load.dump((features, img_files.values))
        return features, img_files.values
    def extract_feature_from_file(self, fname):
        img = cv2.imread(fname)
        img = img[...,::-1]
        return self.extract_features(img)

    def extract_features(self, rgb_img):
        #assume the input image is of RGB
        feature_list = []
        if 'hog' in self.used_features:
            gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
            hog_features = self.get_hog_features(gray)
            feature_list.append(hog_features)
         
        if 'color' in self.used_features:
            hls_image = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HLS)
            _, _, _, _, color_hist_features = self.color_hist(hls_image)
            feature_list.append(color_hist_features)
            
        if 'raw' in self.used_features:
            bs_features = self.bin_spatial(rgb_img)
            feature_list.append(bs_features)
      
        features = np.concatenate(feature_list)
         
        return features
   
       
    def extract_features_labels(self):
        
        dump_load = DumpLoad(self.feature_pickle)
        if dump_load.isExisiting():
            return dump_load.load()
        df = pd.read_csv('../data/label.csv', index_col=0)
        img_files = df['FileName']
        
        features = []
        for fname in img_files.values:
            feature = self.extract_feature_from_file(fname)
            features.append(feature)
       
        features = np.asarray(features)
        labels = df['label'].values.astype(np.int32)
        
        res = (features,labels)
        dump_load.dump(res)   
        return res
   
    def run(self):
        if os.path.exists(self.feature_pickle):
            os.remove(self.feature_pickle)
        self.extract_features_labels()
       
       
       
        plt.show()

        return



if __name__ == "__main__":   
    obj= PreprocessData()
    obj.run()