import sys
import os

sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from postprocess.sliding_window import SlidingWindow
from implement.svmmodel import SVMModel
from utility.vis_utils import visualize_grid,vis_grid
import datetime
from time import time
from postprocess.mergebbox import g_mbbx



class DetectionInImage(SlidingWindow, SVMModel):
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
        roi_sclaed = self.__get_roi(img, sliding_window)
        res = self.predict_sliding_window(roi_sclaed)
#         res = True if res >0 else False
        return res
    def __get_roi(self, img, sliding_window):
        x1,y1 = sliding_window[0]
        x2,y2 = sliding_window[1]
        roi = img[y1:y2,x1:x2]
        roi_sclaed = cv2.resize(roi, (64,64))
        return roi_sclaed
    def save_hard_samples(self,hard_samples_folder, img, car_windows,frame_num):
        if hard_samples_folder is None:
            return
        count = 1
        for car_windows in car_windows:
            roi_sclaed = self.__get_roi(img, car_windows)
            now = datetime.datetime.now()
            now = now.strftime("%Y_%b_%d_%H_%m_%S_%f")
#             fname = hard_samples_folder + now + '_' + str(count) + '_'+ '.png'
            fname = hard_samples_folder + "_".join((str(frame_num), str(count), now)) + '.png'
            mpimg.imsave(fname, roi_sclaed)
            count += 1
        return
    def process_image_RGB(self, img, hard_samples_folder = None, frame_num = None, Debug = False):
        #Get all sliding windows
        sliding_windows = self.get_sliding_windows(img)
        bboxes = []
        bboxes_scores = []
        for sliding_window in sliding_windows:
            is_car = self.check_sliding_window(img, sliding_window)
            if is_car > 0:
                bboxes.append(sliding_window)
                bboxes_scores.append(is_car)
        #initial boudning box
        img_bouding_box = img.copy()
        img_bouding_box = self.draw_boxes(img_bouding_box, bboxes, color=(0, 0, 255), thick=6, bboxes_scores = bboxes_scores)
        if Debug:
            img_bouding_box = self.draw_boxes(img_bouding_box, sliding_windows, color=(255, 255, 255), thick=2)
        
        #image after merging
        bboxes,bboxes_scores = g_mbbx.merge_bbox(img, bboxes,bboxes_scores)    
        img_merged = self.draw_boxes(img, bboxes, color=(0, 0, 255), thick=6, bboxes_scores = bboxes_scores)  
        
        self.save_hard_samples(hard_samples_folder, img, bboxes,frame_num)     
        
        right_side = self.stack_image_vertical([img,img_bouding_box])
        left_side = img_merged
        img_final = self.stack_image_horizontal([left_side, right_side], max_img_width = left_side.shape[1], max_img_height= left_side.shape[0])
                     
        return img_final
    
    
    def run(self):
        t0 = time()
        fnames = []
#         fnames = ['./test_images/straight13.jpg','./test_images/straight14.jpg','./test_images/straight15.jpg',
#                   './test_images/straight16.jpg','./test_images/straight17.jpg']
        fnames_test = ['../data/test_images/test1.jpg','../data/test_images/test2.jpg','../data/test_images/test3.jpg','../data/test_images/test4.jpg',
          '../data/test_images/test5.jpg','../data/test_images/test6.jpg']
        fnames_cars = ['../data/test_images/car0.jpg','../data/test_images/car5.jpg','../data/test_images/car10.jpg','../data/test_images/car15.jpg',
                  '../data/test_images/car20.jpg','../data/test_images/car25.jpg','../data/test_images/car26.jpg','../data/test_images/car27.jpg',
                  '../data/test_images/car28.jpg','../data/test_images/car29.jpg','../data/test_images/car30.jpg','../data/test_images/car32.jpg',
          '../data/test_images/car34.jpg','../data/test_images/car36.jpg','../data/test_images/car48.jpg','../data/test_images/car50.jpg']
        fnames_hardframes = ['../data/hard_frames/frame_0.jpg','../data/hard_frames/frame_187.jpg','../data/hard_frames/frame_266.jpg',
                            '../data/hard_frames/frame_338.jpg','../data/hard_frames/frame_513.jpg','../data/hard_frames/frame_622.jpg','../data/hard_frames/frame_723.jpg',
                            '../data/hard_frames/frame_774.jpg','../data/hard_frames/frame_952.jpg','../data/hard_frames/frame_1041.jpg','../data/hard_frames/frame_1074.jpg',
                            '../data/hard_frames/frame_1206.jpg']
        hard_frames = []
#         fnames = ['./test_images/challenge0.jpg','./test_images/challenge1.jpg','./test_images/challenge2.jpg','./test_images/challenge3.jpg',
#           './test_images/challenge4.jpg','./test_images/challenge5.jpg','./test_images/challenge6.jpg','./test_images/challenge7.jpg']
#         fnames = ['../data/test_images/test4.jpg']
        
        fnames.extend(fnames_hardframes)
#         fnames.extend(fnames_test)
#         fnames.extend(fnames_cars)
#         fnames.extend(fnames_smallcars)
#         fnames = ['../data/test_images/car29.jpg']
        res_imgs = []
        for fname in fnames:
            img = mpimg.imread(fname)
            img_final = self.process_image_RGB(img, None, None,Debug = False)
            res_imgs.append(img)
            
        print("prediction time:", round(time()-t0, 3), "s")
       
            
         
        res_imgs = self.stack_image_vertical(res_imgs)
 
        
        plt.imshow(res_imgs)
        plt.show()
        return



if __name__ == "__main__":   
    obj= DetectionInImage()
    obj.run()