import numpy as np
import cv2
import matplotlib.pyplot as plt
from postprocess.drawboundingbox import DrawBoundingBox


class HeatMap(object):
    def __init__(self):
        return
    def get_bboxes(self, img, bboxes,bboxes_scores):
        bboxes = np.asarray(bboxes)
        bboxes_scores = np.asarray(bboxes_scores)
        heat_map_img = self.__get_heat_map(img, bboxes, bboxes_scores)
        return self.__hot2boundingBox(heat_map_img,img)
    def __get_heat_map(self,img, bboxes,bboxes_scores):
        heat_map_img = np.zeros(img.shape[:2],dtype=np.float32)
        for i in range(len(bboxes_scores)):
            win = bboxes[i]
            heat_map_img[win[1]:win[3],win[0]:win[2]] += bboxes_scores[i]
        return heat_map_img
    def __hot2boundingBox(self,heat_map_img,image, hotThresh=3.5):

        
       
        binary_zeros = np.zeros_like(heat_map_img, dtype=np.uint8)
        #here we assume the maximum is 255
        color_warp = np.dstack((binary_zeros, ((heat_map_img/5.0 )* 255).astype(np.uint8), binary_zeros))
        heat_map = cv2.addWeighted(image, 1, color_warp, 1, 0)
        
        
        heat_map_img[heat_map_img <= hotThresh] = 0
        heat_map_img[heat_map_img > hotThresh] = 255
        plt.imshow(heat_map_img, cmap='gray')
        contours, _ = cv2.findContours(heat_map_img.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#         im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        boundingBox=[]
        bboxes_scores = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
#             cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
#             M = cv2.moments(cnt)
#             cx = int(M['m10'] / M['m00'])
#             cy = int(M['m01'] / M['m00'])
#             centers.append([cx,cy])
            boundingBox.append([x, y,x + w, y + h])
            bboxes_scores.append(1.0)
#             cv2.circle(result, (cx, cy), 10, (0, 0, 255), 5)
        return boundingBox, bboxes_scores, heat_map

class MergeBBox(DrawBoundingBox):
    def __init__(self):
        DrawBoundingBox.__init__(self)
    

        return
    def __filer_low_score_bbox(self,bboxes,bboxes_scores):
        
        cars = bboxes_scores > 0.1

        return bboxes[cars],bboxes_scores[cars]
    
    def __get_bbox_groupRec(self,bboxes_rec ):
#         bboxes_rec2 = np.array([[1,2,3,4],[1,2,3,4],[4,5,6,7],[4,5,6,7]] )
        bboxes_rec,bboxes_scores = cv2.groupRectangles(bboxes_rec, 2, 0.2)
        if len(bboxes_scores) != 0:
            bboxes_scores = bboxes_scores.ravel()
        return bboxes_rec,bboxes_scores
    def merge_bbox(self, img, bboxes,bboxes_scores):
        if len(bboxes_scores) == 0:
            return img,img
        img_all_boxes = self.draw_boxes(img, bboxes, color=(0, 0, 255), thick=6, bboxes_scores = bboxes_scores) 
        bboxes,bboxes_scores = self.__filer_low_score_bbox(bboxes, bboxes_scores)
        img_filtered_boxes = self.draw_boxes(img, bboxes, color=(0, 0, 255), thick=6, bboxes_scores = bboxes_scores)
#         if len(bboxes_scores) == 0:
#             return img, img_filtered_boxes
         
        bboxes,bboxes_scores = self.__get_bbox_groupRec(bboxes.tolist())
        img_merged_boxes = self.draw_boxes(img, bboxes, color=(0, 0, 255), thick=6, bboxes_scores = bboxes_scores)
        
#         
#         bboxes=[]
#         print("merged bouding boxes")
#         for i in range(len(bboxes_scores)):
#             bbox_rec = bboxes_rec[i]
#             x1,y1,x2,y2 = bbox_rec
#             score = bboxes_scores[i]
#             size = (x2-x1,y2-y1)
#             
#             print("size {}: score {:.2f}, pos{}".format(size, score, ((x1,y1),(x2,y2 ))))
#             bboxes.append(((x1,y1),(x2,y2 )))
         
        return img_all_boxes,img_filtered_boxes,img_merged_boxes
    
    

   
   
    def run(self):
        res = self.get_sliding_window_configs()

        return

g_mbbx = MergeBBox()

if __name__ == "__main__":   
    obj= MergeBBox()
    obj.run()