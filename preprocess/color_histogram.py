import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class ColorHistogram(object):
    def __init__(self):
        return
    # Define a function to compute color histogram features  
    def color_hist(self, img, nbins=32, bins_range=(0, 256)):
        # Compute the histogram of the RGB channels separately
        rhist = np.histogram(img[:,:,0], bins=32, range=(0, 256))
        ghist = np.histogram(img[:,:,1], bins=32, range=(0, 256))
        bhist = np.histogram(img[:,:,2], bins=32, range=(0, 256))
        # Generating bin centers
        bin_edges = rhist[1]
        bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
        # Return the individual histograms, bin_centers and feature vector
        return rhist, ghist, bhist, bin_centers, hist_features

   
    def run(self):
        fname = '../data/cutout1.jpg'
        image = mpimg.imread(fname)
       
        rh, gh, bh, bincen, feature_vec = self.color_hist(image, nbins=32, bins_range=(0, 256))
        if rh is not None:
            fig = plt.figure(figsize=(12,3))
            plt.subplot(131)
            plt.bar(bincen, rh[0])
            plt.xlim(0, 256)
            plt.title('R Histogram')
            plt.subplot(132)
            plt.bar(bincen, gh[0])
            plt.xlim(0, 256)
            plt.title('G Histogram')
            plt.subplot(133)
            plt.bar(bincen, bh[0])
            plt.xlim(0, 256)
            plt.title('B Histogram')
            fig.tight_layout()
        plt.show()

        return



if __name__ == "__main__":   
    obj= ColorHistogram()
    obj.run()