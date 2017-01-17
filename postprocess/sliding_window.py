import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from postprocess.drawboundingbox import DrawBoundingBox
from utility.vis_utils import vis_grid



class SlidingWindow(DrawBoundingBox):
    def __init__(self):
        DrawBoundingBox.__init__(self)
        return
    
    # Define a function that takes an image,
    # start and stop positions in both x and y, 
    # window size (x and y dimensions),  
    # and overlap fraction (for both x and y)
    def __get_slide_windows(self, img, x_start_stop=[None, None], y_start_stop=[None, None], 
                        xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
        # If x and/or y start/stop positions not defined, set to image size
        if x_start_stop[0] == None:
            x_start_stop[0] = 0
        if x_start_stop[1] == None:
            x_start_stop[1] = img.shape[1]
        if y_start_stop[0] == None:
            y_start_stop[0] = 0
        if y_start_stop[1] == None:
            y_start_stop[1] = img.shape[0]
        # Compute the span of the region to be searched    
        xspan = x_start_stop[1] - x_start_stop[0]
        yspan = y_start_stop[1] - y_start_stop[0]
        # Compute the number of pixels per step in x/y
        nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
        ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
        # Compute the number of windows in x/y
        nx_windows = np.int(xspan/nx_pix_per_step) - 1
        ny_windows = np.int(yspan/ny_pix_per_step) - 1
        # Initialize a list to append window positions to
        window_list = []
        # Loop through finding x and y window positions
        # Note: you could vectorize this step, but in practice
        # you'll be considering windows one by one with your
        # classifier, so looping makes sense
        for ys in range(ny_windows):
            for xs in range(nx_windows):
                # Calculate window position
                startx = xs*nx_pix_per_step + x_start_stop[0]
                endx = startx + xy_window[0]
                starty = ys*ny_pix_per_step + y_start_stop[0]
                endy = starty + xy_window[1]
                # Append window position to list
                window_list.append(((startx, starty), (endx, endy)))
        # Return the list of windows
        return window_list
    def get_sliding_windows(self, img):

        window_configs = []
        window_configs.append(((200, 160),[None, None], [420, 700] ,(0.5, 0.5)))
#         window_configs.append(((80*2, 60*2),[None, None], [320+80, 320 + 80*4] ,(0.5, 0.5)))
         
#         window_configs.append(((40*2, 40*2),[None, None], [320+40, 320+ 40*4] ,(0.5, 0.5)))
#         window_configs.append(((55*2, 55*2),[None, None], [320+55, 320+ 55*4] ,(0.5, 0.5)))
#          
#         window_configs.append(((50*2, 40*2),[None, None], [320+50, 320+ 50*4] ,(0.5, 0.5)))
        
#         window_configs.append(((10*2, 10*2),[None, None], [405, 440] ,(0.5, 0.5)))
        


         

        windows = []
        for window_config in window_configs:
            xy_window, x_start_stop, y_start_stop,xy_overlap = window_config
            cur_windows = self.__get_slide_windows(img, x_start_stop=x_start_stop, y_start_stop=y_start_stop, 
                        xy_window=xy_window, xy_overlap=xy_overlap)
            windows.extend(cur_windows)
        return windows
   
   
    def run(self):
        fnames = []
        #         fnames = ['./test_images/straight13.jpg','./test_images/straight14.jpg','./test_images/straight15.jpg',
#                   './test_images/straight16.jpg','./test_images/straight17.jpg']
        fnames_test = ['../data/test_images/test1.jpg','../data/test_images/test2.jpg','../data/test_images/test3.jpg','../data/test_images/test4.jpg',
          '../data/test_images/test5.jpg','../data/test_images/test6.jpg']
        fnames_cars = ['../data/test_images/car0.jpg','../data/test_images/car5.jpg','../data/test_images/car10.jpg','../data/test_images/car15.jpg',
                  '../data/test_images/car20.jpg','../data/test_images/car25.jpg','../data/test_images/car26.jpg','../data/test_images/car27.jpg',
                  '../data/test_images/car28.jpg','../data/test_images/car29.jpg','../data/test_images/car30.jpg','../data/test_images/car32.jpg',
        '../data/test_images/car34.jpg','../data/test_images/car36.jpg','../data/test_images/car48.jpg','../data/test_images/car50.jpg','../data/hard_frames/frame_1108.jpg']
        
#         fnames = ['./test_images/challenge0.jpg','./test_images/challenge1.jpg','./test_images/challenge2.jpg','./test_images/challenge3.jpg',
#           './test_images/challenge4.jpg','./test_images/challenge5.jpg','./test_images/challenge6.jpg','./test_images/challenge7.jpg']
        
        
        fnames.extend(fnames_test)
        fnames.extend(fnames_cars)
        fnames = ['../data/test_images/car50.jpg']
        res_imgs = []
        for fname in fnames:
            img = mpimg.imread(fname)
            # Add bounding boxes in this format, these are just example coordinates.
            windows = self.get_sliding_windows(img)       
            window_img = self.draw_boxes(img, windows, color=(0, 0, 255), thick=6)     
            res_imgs.append(window_img)
            
        
        res_imgs = np.asarray(res_imgs)
        res_imgs = vis_grid(res_imgs)
                       
        plt.imshow(res_imgs)

        plt.show()

        return



if __name__ == "__main__":   
    obj= SlidingWindow()
    obj.run()