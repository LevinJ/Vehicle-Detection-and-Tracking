import sys
import os
sys.path.insert(0, os.path.abspath('..')) 

from utility.dumpload import DumpLoad
import cv2



class AnalyzePrediction():    

    def __init__(self):
        self.refPt = []
        self.cropping = False

        return
    def __load_model(self):
        dump_load = DumpLoad('../data/smvmodel.pickle')
        self.model = dump_load.load()

        return
    def click_and_crop(self, event, x, y, flags, param):
        # grab references to the global variables

     
        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being
        # performed
        if event == cv2.EVENT_LBUTTONDOWN:
            self.refPt = [(x, y)]
            self.cropping = True
     
        # check to see if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            # record the ending (x, y) coordinates and indicate that
            # the cropping operation is finished
            self.refPt.append((x, y))
            self.cropping = False
     
            # draw a rectangle around the region of interest
            cv2.rectangle(self.image, self.refPt[0], self.refPt[1], (0, 255, 0), 2)
            self.refPt = []
        elif event == cv2.EVENT_MOUSEMOVE:
            print(x,y)
            if not self.cropping:
                return
            print('temp image')
            self.image = self.clone.copy()
            cv2.rectangle(self.image, self.refPt[0], (x,y), (255, 0, 0), 2)

    def predict_roi(self, img_path):
        self.image = cv2.imread(img_path)
        self.clone = self.image.copy()
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", self.click_and_crop)
        
        # keep looping until the 'q' key is pressed
        while True:
            # display the image and wait for a keypress
            cv2.imshow("image", self.image)
            key = cv2.waitKey(1) & 0xFF
         
            # if the 'r' key is pressed, reset the cropping region
            if key == ord("r"):
                self.image = self.clone.copy()
         
            # if the 'c' key is pressed, break from the loop
            elif key == ord("c"):
                break
        return
    
    
    def run(self):
        self.__load_model()
        img_path = '../data/test_images/test1.jpg'
        self.predict_roi(img_path)
        
        
        


        return
    


if __name__ == "__main__":   
    obj= AnalyzePrediction()
    obj.run()