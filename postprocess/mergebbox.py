import numpy as np
import cv2
import matplotlib.pyplot as plt
from postprocess.drawboundingbox import DrawBoundingBox
from skimage.feature import blob_doh
from sklearn.cluster import DBSCAN
from sklearn import metrics
import matplotlib.pyplot as plt
from postprocess.frametracking import g_frame_tracking


class HeatMap(DrawBoundingBox):
    def __init__(self):
        DrawBoundingBox.__init__(self)
        return
    def get_bboxes(self, img, bboxes,bboxes_scores):
        heat_map_img = self.__get_heat_map(img, bboxes, bboxes_scores)
        heat_map_before_thres = heat_map_img.copy()
        #threshold the heat heat map
        thres = 80
        heat_map_img [heat_map_img < thres] = 0
        heat_map_img [heat_map_img >= thres] = 255
        if ((heat_map_img==255).sum()==0):
            return heat_map_img,heat_map_before_thres
            
#         plt.imshow(heat_map_img[:,:,0], cmap="gray")
        
     
        blobs = blob_doh(heat_map_img[:,:,0], min_sigma=30, max_sigma=150, num_sigma=2, threshold=0.01, overlap=0.2, log_scale=False)
        for blob in blobs:
            y, x, r = blob
            cv2.circle(heat_map_img, (int(x),int(y)),int(r),(255,0,0),thickness=6)

        print(blobs)
        return heat_map_img,heat_map_before_thres
    
    def __get_heat_map(self,img, bboxes,bboxes_scores):
#         heat_map_img = img.copy()
        heat_map_img = np.zeros_like(img)
        
        for i in range(len(bboxes_scores)):
            win = bboxes[i]
            heat_map_img[win[1]:win[3],win[0]:win[2]] += 40
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
class Clustering(DrawBoundingBox):
    def __init__(self):
        DrawBoundingBox.__init__(self)
        return 
    def get_bboxes(self, img, bboxes,bboxes_scores):
        if len(bboxes) == 0:
            return img, img,img,[],[]

        labels,centers,clustering_img_temp = self.__cluster_bdboxes(img, bboxes,bboxes_scores)
        
        clustering_img = self.__draw_clustering(img, labels, bboxes, centers)
        
        
        #display clustering result
        unique_labels = np.unique(labels)
#         colors = [(255,0,0),(0,255,0),(125,125,0),(0,125,125),(125,0,125),(0,0,125),(125,0,0)]
        merged_boxes = []
        merged_boxes_scores = []
        for i in range(len(unique_labels)):
            label = unique_labels[i]
            group_indx = labels == label
            if label == -1:
                #this is a noisy detection, skip it
                continue
            self.__add_outer_box(bboxes, bboxes_scores, group_indx, merged_boxes, merged_boxes_scores,clustering_img)
            

            
        #draw the merged merged box    
        merged_img = img.copy()
        merged_img = self.draw_boxes(merged_img, merged_boxes)
    
        return clustering_img_temp, clustering_img,merged_img,merged_boxes,merged_boxes_scores
    def __get_bdbox_centers(self, bboxes):
        center_xs = (bboxes[:,0]+bboxes[:,2])/2
        center_ys = (bboxes[:,1]+bboxes[:,3])/2
        centers =np.concatenate((center_xs[:,np.newaxis], center_ys[:,np.newaxis]), axis = 1).astype(np.int16)
        return centers
    def __draw_clustering(self, img, labels, bboxes, centers):
        #display clustering result
        clustering_img = np.zeros_like(img)
        unique_labels = np.unique(labels)
        colors = [(255,0,0),(0,255,0),(125,125,0),(0,125,125),(125,0,125),(0,0,125),(125,0,0)]
        for i in range(len(unique_labels)):
            label = unique_labels[i]
            if label == -1:
                # white used for noise.
                color = (255,255,255)
            else:
                color = colors[i]
            group_indx = labels == label
            for bbox, center in zip(bboxes[group_indx],centers[group_indx] ):
                pt1,pt2 = tuple(bbox[:2]), tuple(bbox[2:])
                cv2.rectangle(clustering_img, pt1, pt2, color=color, thickness=6)
                cv2.circle(clustering_img, tuple(center), 4,color,thickness=-1)
        return clustering_img
    def __cluster_bdboxes(self, img, bboxes,bboxes_scores):
        centers = self.__get_bdbox_centers(bboxes)
        
        

        db = DBSCAN(eps=100, min_samples=2).fit(centers)
        
        

        labels = db.labels_
        clustering_img_temp = self.__draw_clustering(img, labels, bboxes, centers)
        labels = self.__reclustering(labels, img, bboxes, bboxes_scores, centers)
           
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        
        print('Estimated number of clusters: %d' % n_clusters_)
        return labels,centers, clustering_img_temp
    def __reclustering(self,labels, img, bboxes,bboxes_scores,centers):
        unique_labels = set(labels)
        # do one more clutering if the blob is too large
        for label in unique_labels:
            if label == -1:
                continue
            group_sel = label == labels
            if group_sel.sum() < 10:
                continue
             
            group_indices = group_sel.nonzero()[0]
            
            sub_sel = bboxes_scores[group_sel] > 0.7
            sub_indices = sub_sel.nonzero()[0]
            SUB_MIN_NUM = 3
            if sub_sel.sum() <= SUB_MIN_NUM*2:
                continue
            sub_bboxes = bboxes[group_sel][sub_sel]
            
            #set all the points in this group as noise first
            group_labels = np.full(len(group_indices), 0, dtype = np.int16)
            group_labels[(~sub_sel)] = -1
            
            sub_centers = self.__get_bdbox_centers(sub_bboxes)
                
            sub_db = DBSCAN(eps=80, min_samples=SUB_MIN_NUM).fit(sub_centers)
            sub_labels = sub_db.labels_
            sub_labels = label * 100 + sub_labels
            group_labels[sub_indices] = sub_labels
            labels[group_indices] = group_labels
        return labels
    def __is_in_reaonable_region(self,width,height,center):
        _,y = center
        if (width< 100 and height < 100):
            if y>500:
                print("rejected: height {}, width {}, center{}".format(width, height, center))
                return False
        if (y < 405 and width >= 72):
            #remove detections that are unreasonalby larege near the top
            return False
        
        return True
    def __add_outer_box(self, bboxes, bboxes_scores, group_indx,merged_boxes,merged_boxes_scores,heat_map_img):
        
        x1 =  bboxes[group_indx][:,0].min()
        y1 =  bboxes[group_indx][:,1].min()
        x2 = bboxes[group_indx][:,2].max()
        y2 = bboxes[group_indx][:,3].max()
        width = (x2-x1)
        height = (y2-y1)
        center = ((x1+x2)/2, (y1+y2)/2)
        if not self.__is_in_reaonable_region(width,height,center):
            return
        
        if len(bboxes)> 3:
#             x1 = int(x1 + height/5.0)
#             x2 = int(x2 - height/5.0)
            y1 = int(y1 + height/5.0)
            y2 = int(y2 - height/5.0)
        merged_box = [x1,y1,x2,y2]
        merged_boxes.append(merged_box)
        merged_boxes_scores.append(bboxes_scores[group_indx])
        pt1,pt2 = tuple(merged_box[:2]), tuple(merged_box[2:])
        cv2.rectangle(heat_map_img, pt1, pt2, color=(0,0,255), thickness=6)
        return


class MergeBBox(DrawBoundingBox):
    def __init__(self):
        DrawBoundingBox.__init__(self)
    

        return
    def __filer_low_score_bbox(self,bboxes,bboxes_scores):
        
        cars = bboxes_scores > 0.1

        return bboxes[cars],bboxes_scores[cars]
    
    def __get_bbox_groupRec(self,bboxes_rec ):
        bboxes_rec,bboxes_scores = cv2.groupRectangles(bboxes_rec, 2, 0.2)
        if len(bboxes_scores) != 0:
            bboxes_scores = bboxes_scores.ravel()
        return bboxes_rec,bboxes_scores
    def merge_bbox(self, img, bboxes,bboxes_scores,frame_num):
        if len(bboxes_scores) == 0:
            return img,img,np.zeros_like(img),np.zeros_like(img),img, img,np.zeros_like(img)
        img_all_boxes = self.draw_boxes(img, bboxes, color=(0, 0, 255), thick=6, bboxes_scores = bboxes_scores) 
        bboxes,bboxes_scores = self.__filer_low_score_bbox(bboxes, bboxes_scores)
        img_filtered_boxes = self.draw_boxes(img, bboxes, color=(0, 0, 255), thick=6, bboxes_scores = bboxes_scores)

#         bboxes,bboxes_scores = self.__get_bbox_groupRec(bboxes.tolist())
        
#         img_merged_boxes = self.draw_boxes(img, bboxes, color=(0, 0, 255), thick=6, bboxes_scores = bboxes_scores)
#         heat_map_img,heat_map_before_thres = HeatMap().get_bboxes(img, bboxes,bboxes_scores)
        clustering_img_temp, clustering_img,merged_img,merged_boxes,merged_boxes_scores = Clustering().get_bboxes(img, bboxes,bboxes_scores)
        
        checked_boxes,heat_map = g_frame_tracking.check_cars(merged_boxes,merged_boxes_scores,frame_num)
        #draw the checked image
        checked_img = img.copy()
        checked_img = self.draw_boxes(checked_img, checked_boxes)

         
        return img_all_boxes,img_filtered_boxes,clustering_img_temp, clustering_img,merged_img,checked_img,heat_map
    
    

   
   
    def run(self):
        res = self.get_sliding_window_configs()

        return

g_mbbx = MergeBBox()

if __name__ == "__main__":   
    obj= MergeBBox()
    obj.run()