import numpy as np
import cv2




class MergeBBox(object):
    def __init__(self):
        sliding_windows = {}
        sliding_windows[(64,64)]= {'config': (([None, None],[330, 720], [32, 32])), 
                                   'thres': 0.0}
        sliding_windows[(128,128)]= {'config': (([None, None],[330, 720], [32, 32])), 
                                   'thres': 0.0}
#         sliding_windows[(192,192)]= {'config': (([None, None],[330, 720], [32, 32])), 
#                                    'thres': 0.0}
        self.sliding_windows = sliding_windows
        self.sliding_windows_config = self.parse_sliding_window_configs()

        return
    def parse_sliding_window_configs(self):
        config = []
        
        for size, v in self.sliding_windows.iteritems():
            item = []
            item.append(size)
            item .extend(v['config'])
            config.append(item)     
        return config
    def __filer_low_score_bbox(self,bboxes,bboxes_scores):
        new_bboxes = []
        new_bboxes_scores = []
        for i in range(len(bboxes_scores)):
            bbox = bboxes[i]
            pt1,pt2 = bbox
            size = (pt2[0]-pt1[0], pt2[1] - pt1[1])
            score = bboxes_scores[i]
            thres = self.sliding_windows[size]['thres']
            print("size {}: score {:.2f}, pos{}".format(size, score, bbox))
            if score > thres:
                new_bboxes.append(bbox)
                new_bboxes_scores.append(score)
                
        return new_bboxes,new_bboxes_scores
    def merge_bbox(self, img, bboxes,bboxes_scores):
        if len(bboxes_scores) == 0:
            return bboxes,bboxes_scores 
        bboxes,bboxes_scores = self.__filer_low_score_bbox(bboxes, bboxes_scores)
        bboxes_rec = []   
        for bbox in bboxes:
            bboxes_rec.append([item for pt in bbox for item in pt])
        bboxes_rec,bboxes_scores = cv2.groupRectangles(bboxes_rec, 1, 0.2)
        bboxes_scores = bboxes_scores.ravel()
        bboxes=[]
        for bbox_rec in bboxes_rec:
            x1,y1,x2,y2 = bbox_rec
            bboxes.append(((x1,y1),(x2,y2 )))
         
        return bboxes,bboxes_scores
    
    

   
   
    def run(self):
        res = self.get_sliding_window_configs()

        return

g_mbbx = MergeBBox()

if __name__ == "__main__":   
    obj= MergeBBox()
    obj.run()