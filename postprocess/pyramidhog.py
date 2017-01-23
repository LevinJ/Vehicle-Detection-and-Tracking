from preprocess.preprocessdata  import PreprocessData
import cv2

class PyramidHog(PreprocessData):
    def __init__(self):
        PreprocessData.__init__(self)
        return 
    def __get_window_feature_one_scale(self, rgb_img, scale, top_clip = 0, step=1):
        #get widnows/features for a particular scale
        hog_arr = self.extract_lab_hog_mulit_dimenstion(rgb_img[top_clip:])
        _,block_y_dim, block_x_dim, _,_,_ = hog_arr.shape
        
        # our sliding widnows of size 64x64 will move across the image one block with one cell stride, which is 8 pixels
        # one slideing widnwos will occupy 8 cells, we will exclude edge of the image when it can not fit into the sliding window
        for y in range(0, block_y_dim -7, step):
            for x in range(0, block_x_dim-7, step):
                top_x = x*8
                top_y = top_clip + y*8
                bottom_x = (x + 8) * 8
                bottom_y = top_clip + (y+8)*8
                
                window = [top_x,top_y,bottom_x,bottom_y] # the position of sliding window
                feature = hog_arr[:,y:y+7,x:x+7,:,:,:] # hog feature for this sliding window
                
                
      
        return window, feature
    def get_window_feature(self, rgb_img):
        scales = [1.0, 1.3, 1.7, 2.2, 2.9]
        
        bdboxes = []
        confidences = []
        for scale in scales:
            top_clip = int(350 / scale)
            step = 1
            img_resized = cv2.resize(rgb_img, (int(rgb_img.shape[1] / scale), int(rgb_img.shape[0] / scale)),
                                    interpolation=cv2.INTER_CUBIC)

            bdbox, confidence = self.__predict_one_scale(img_resized, scale, top_clip, step)
            bdboxes.extend(bdbox)
            confidences.extend(confidence)
        return