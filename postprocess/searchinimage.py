import sys
import os

sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from postprocess.sliding_window import SlidingWindow
from implement.svmmodel import SVMModel



class SearchInImage(SlidingWindow, SVMModel):
    def __init__(self):
        SlidingWindow.__init__(self)
        SVMModel.__init__(self)
        return
    def stack_image_horizontal(self, imgs, max_img_width = None, max_img_height=None):
        return self.__stack_image_horizontal(imgs, axis = 1, max_img_width = max_img_width, max_img_height=max_img_height)
    def stack_image_vertical(self, imgs, max_img_width = None, max_img_height=None):
        return self.__stack_image_horizontal(imgs, axis = 0, max_img_width = max_img_width, max_img_height=max_img_height)
    def __stack_image_horizontal(self, imgs, axis = 1, max_img_width = None, max_img_height=None):
        #first let's make sure all the imge has same size
        img_sizes = np.empty([len(imgs), 2], dtype=int)
        for i in range(len(imgs)):
            img = imgs[i]
            img_sizes[i] = np.asarray(img.shape[:2])
        if max_img_width is None:
            max_img_width = img_sizes[:,1].max()
        if max_img_height is None:
            max_img_height = img_sizes[:,0].max()
        for i in range(len(imgs)):
            img = imgs[i]
            img_width = img.shape[1]
            img_height = img.shape[0]
            if (img_width == max_img_width) and (img_height == max_img_height):
                continue
            imgs[i] = cv2.resize(img, (max_img_width,max_img_height))
            
            
        
        for i in range(len(imgs)):
            img = imgs[i]
            if len(img.shape) == 2:
                scaled_img = np.uint8(255*img/np.max(img))
                imgs[i] = cv2.cvtColor(scaled_img, cv2.COLOR_GRAY2BGR)
#                 plt.imshow(imgs[i][...,::-1])
        res_img = np.concatenate(imgs, axis=axis)
        return res_img
    def check_sliding_window(self,img, sliding_window):
        x1,y1 = sliding_window[0]
        x2,y2 = sliding_window[1]
        roi = img[y1:y2,x1:x2]
        roi_sclaed = cv2.resize(roi, (64,64))
        res = self.predict_sliding_window(roi_sclaed)
        return res
    def detect_cars(self, img):
        #Get all sliding windows
        sliding_windows = self.get_sliding_windows(img)
        car_windows = []
        for sliding_window in sliding_windows:
            is_car = self.check_sliding_window(img, sliding_window)
            if is_car:
                car_windows.append(sliding_window)
        window_img = self.draw_boxes(img, car_windows, color=(0, 0, 255), thick=6)                    
        return window_img
    
    
    def run(self):
#         fnames = ['./test_images/straight13.jpg','./test_images/straight14.jpg','./test_images/straight15.jpg',
#                   './test_images/straight16.jpg','./test_images/straight17.jpg']
        fnames = ['./test_images/test1.jpg','./test_images/test2.jpg','./test_images/test3.jpg','./test_images/test4.jpg',
          './test_images/test5.jpg','./test_images/test6.jpg']
#         fnames = ['./test_images/challenge0.jpg','./test_images/challenge1.jpg','./test_images/challenge2.jpg','./test_images/challenge3.jpg',
#           './test_images/challenge4.jpg','./test_images/challenge5.jpg','./test_images/challenge6.jpg','./test_images/challenge7.jpg']
#         fnames = ['./test_images/challenge2.jpg']
        fnames = ['../data/test_images/test5.jpg']
        

        res_imgs = []
        for fname in fnames:
            img = mpimg.imread(fname)
            img = self.detect_cars(img)
            res_imgs.append(img)
         
        res_imgs = self.stack_image_vertical(res_imgs)
 
        
        plt.imshow(res_imgs)
        plt.show()
        return



if __name__ == "__main__":   
    obj= SearchInImage()
    obj.run()