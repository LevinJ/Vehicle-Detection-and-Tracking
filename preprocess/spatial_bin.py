import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class SpatialBin(object):
    def __init__(self):
        return
    
    # Define a function to compute color histogram features  
    # Pass the color_space flag as 3-letter all caps string
    # like 'HSV' or 'LUV' etc.
    def bin_spatial(self, img, color_space='RGB', size=(32, 32)):
        # Convert image to new color space (if specified)
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        else: feature_image = np.copy(img)             
        # Use cv2.resize().ravel() to create the feature vector
        features = cv2.resize(feature_image, size).ravel() 
        return features
    def run(self):
        fname = '../data/cutout1.jpg'
        image = mpimg.imread(fname)
        feature_vec = self.bin_spatial(image, color_space='RGB', size=(32, 32))

        # Plot features
        plt.plot(feature_vec)
        plt.title('Spatially Binned Features')
       
        
        plt.show()

        return



if __name__ == "__main__":   
    obj= SpatialBin()
    obj.run()