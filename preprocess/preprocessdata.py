import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from preprocess.spatial_bin import SpatialBin
from preprocess.color_histogram import ColorHistogram



class PreprocessData(SpatialBin, ColorHistogram):
    def __init__(self):
        SpatialBin.__init__(self)
        ColorHistogram.__init__(self)
        return
    # Define a function to compute color histogram features  
    def __extract_feature(self, img):
        #expect img to be RGB
        sb_features = self.bin_spatial(img, color_space='RGB', size=(32, 32))
        rhist, ghist, bhist, bin_centers, hist_features = self.color_hist(img, nbins=32, bins_range=(0, 256))
        
        feature_list = [sb_features[:,np.newaxis], hist_features[:,np.newaxis]]
        combined_features = np.vstack(feature_list).astype(np.float64)
        return combined_features.squeeze()
       
    def extract_features(self, img_files):
        features = []
        for fname in img_files:
            img = mpimg.imread(fname)
            feature = self.__extract_feature(img)
            features.append(feature)
            
        return np.asanyarray(features)
   
    def run(self):
        img_files = ['../data/cutout1.jpg']
        res = self.extract_features(img_files)
       
       
       
        plt.show()

        return



if __name__ == "__main__":   
    obj= PreprocessData()
    obj.run()