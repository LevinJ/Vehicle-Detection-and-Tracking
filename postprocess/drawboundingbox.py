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
    def draw_boxes(self, img, bboxes, color=(0, 0, 255), thick=6, bboxes_scores = []):
        # make a copy of the image
        draw_img = np.copy(img)
        # draw each bounding box on your image copy using cv2.rectangle()
        # return the image copy with boxes drawn
#         for bbox in bboxes:
        for i in range(len(bboxes)):
            bbox = bboxes[i]
            pt1,pt2 = tuple(bbox[:2]), tuple(bbox[2:])
            cv2.rectangle(draw_img, pt1, pt2, color=color, thickness=thick)
            if len(bboxes_scores) != 0:
                score = bboxes_scores[i]
                x = pt1[0]
                y = int((pt1[1]+ pt2[1])/2.0)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(draw_img,"{:.2f}".format(score),(x,y), font, 1,(0,255,255),2)
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