import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from skimage.feature import hog
from skimage import color, exposure
# images are divided up into vehicles and non-vehicles


class HOGFeature(object):
    def __init__(self):
        return
    # Define a function to return HOG features and visualization
    def get_hog_features(self, img, orient=9, pix_per_cell=8, cell_per_block=2, 
                            vis=False, feature_vec=True):
        # Call with two outputs if vis==True
        if vis == True:
            features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                      cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                                      visualise=vis, feature_vector=feature_vec)
            return features, hog_image
        # Otherwise call with one output
        else:      
            features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                           cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                           visualise=vis, feature_vector=feature_vec)
            return features
   
    def run(self):
        fname = '../data/3.png'
        img = mpimg.imread(fname)
        if len(img.shape)!=2:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        features, hog_image = self.get_hog_features(img, vis=True,feature_vec=False)

        # Plot features
        plt.imshow(hog_image, cmap='gray')
        plt.title('Spatially Binned Features')
       
        
        plt.show()

        return



if __name__ == "__main__":   
    obj= HOGFeature()
    obj.run()