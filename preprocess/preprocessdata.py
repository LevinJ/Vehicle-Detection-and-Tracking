import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from preprocess.spatial_bin import SpatialBin
from preprocess.color_histogram import ColorHistogram
import pandas as pd
from preprocess.hogfeature import HOGFeature



class PreprocessData(SpatialBin, ColorHistogram, HOGFeature):
    def __init__(self):
        SpatialBin.__init__(self)
        ColorHistogram.__init__(self)
        HOGFeature.__init__(self)
        return
    # Define a function to compute color histogram features  
    def __extract_feature(self, img):
        #expect img to be RGB
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        hog_features = self.get_hog_features(gray)
        
#         sb_features = self.bin_spatial(img, color_space='RGB', size=(32, 32))
#         _, _, _, _, hist_features = self.color_hist(img, nbins=32, bins_range=(0, 256))
        
        feature_list = []
#         feature_list.append(sb_features[:,np.newaxis])
#         feature_list.append(hist_features[:,np.newaxis])
        feature_list.append(hog_features[:,np.newaxis])
        
        combined_features = np.vstack(feature_list).astype(np.float64)
        return combined_features.squeeze()
       
    def extract_features(self):
        df = pd.read_csv('../data/label.csv')
        img_files = df['FileName']
        features = []
        for fname in img_files:
            img = mpimg.imread(fname)
            feature = self.__extract_feature(img)
            features.append(feature)
            
        return np.asanyarray(features)
   
    def run(self):
        
        res = self.extract_features()
       
       
       
        plt.show()

        return



if __name__ == "__main__":   
    obj= PreprocessData()
    obj.run()