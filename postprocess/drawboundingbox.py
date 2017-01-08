import sys
import os
# from _pickle import dump
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import cv2
import matplotlib.pyplot as plt



class DrawBoundingBox(object):
    def __init__(self):
        return
    def draw_boxes(self, img, bboxes, color=(0, 0, 255), thick=6):
        # make a copy of the image
        draw_img = np.copy(img)
        # draw each bounding box on your image copy using cv2.rectangle()
        # return the image copy with boxes drawn
        for bbox in bboxes:
            pt1,pt2 = bbox
            cv2.rectangle(draw_img, pt1, pt2, color=color, thickness=thick)
        return draw_img # Change this line to return image copy with boxes
   
    def run(self):
#         fname = './test_images/test1.jpg'
        fname = '../test1.jpg'
        image = cv2.imread(fname)
        # Add bounding boxes in this format, these are just example coordinates.
        bboxes = [((100, 100), (200, 200)), ((300, 300), (400, 400))]
        
        result = self.draw_boxes(image, bboxes)
        result = result[...,::-1] #convert from opencv bgr to standard rgb
        plt.imshow(result)
        plt.show()

        return



if __name__ == "__main__":   
    obj= DrawBoundingBox()
    obj.run()