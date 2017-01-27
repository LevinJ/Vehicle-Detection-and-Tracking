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
        fnames = ['../data/dataset/non-vehicles/Extras/extra5590.png', '../data/dataset/vehicles/GTI_MiddleClose/image0113.png']
        hog_imgs = []
        for fname in fnames:
            img = cv2.imread(fname)
            rgb_img = img[...,::-1]
            one_row = []
            one_row.append(rgb_img)
            lab_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
            for channel in range(3):
                img = lab_img[:,:,channel]
                _, hog_image = self.get_hog_features(img, vis=True,feature_vec=False)
                one_row.append(hog_image)
            hog_imgs.append(one_row)

        # Plot features
        _,axes = plt.subplots(2,4)
        for i in range(len(axes)):
            titles = ['original image', 'L Channel HOG', 'A Channel HOG', 'B Channel HOG']
            for j in range(len(axes[0])):
                if j == 0:
                    axes[i,j].imshow(hog_imgs[i][j])
                else:
                    axes[i,j].imshow(hog_imgs[i][j], cmap='gray')
                axes[i,j].set_title(titles[j])
               
       
        
        plt.show()

        return



if __name__ == "__main__":   
    obj= HOGFeature()
    obj.run()