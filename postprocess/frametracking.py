
import sys
import os
from scipy.constants.constants import carat
from partd.utils import frame
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define a class to receive the characteristics of each line detection

    


class FrameTracking():
    def __init__(self):
        self.enable_tracking = False
        self.df = pd.DataFrame()
       
        return

    def check_cars(self, detected_cars, detected_scores,frame_num):
        if not self.enable_tracking:
            return detected_cars,np.zeros((720,1280,3), dtype=np.uint8)
        if len(detected_cars)== 0:
            return detected_cars,np.zeros((720,1280,3), dtype=np.uint8)
   
 
        heat_map = self.__draw_heat_map(frame_num)
   
        bdboxes = []
        for car, scores in zip(detected_cars, detected_scores):
            x1,y1,x2,y2 = car
            car_info = {}
            car_info['width'] = x2-x1
            car_info['height'] = y2-y1
            car_info['center_x'] = int((x1+x2)/2.0)
            car_info['center_y'] = int((y1 + y2)/2.0)
            car_info['scores'] = scores
            car_info['bdbox'] = car
            car_info['frame_num'] = frame_num
            print("height {}, width {}, center {}, bdbox {}".format(car_info['width'], car_info['height'], 
                                                         (car_info['center_x'],car_info['center_y']),
                                                         car_info['bdbox']))
            car_info['iscar'] = self.__is_car(car_info,heat_map)
            
            
            if (car_info['iscar']):
                adjusted_bdbox = self.__adjust_car_bdbox(frame_num, car_info)
                car_info['adjusted_bdbox'] = adjusted_bdbox
                bdboxes.append(adjusted_bdbox)   
            self.df = self.df.append(car_info, ignore_index=True)
        return np.asarray(bdboxes),heat_map
    def __adjust_car_bdbox(self, frame_num,car_info):
        if len(self.df) == 0: 
            return car_info['bdbox']
        last_frame_num  = frame_num - 10
        conditon_1 = (self.df['frame_num'] >= last_frame_num) & (self.df['iscar'] == True)
        df = self.df[conditon_1]
        condition_2 = ((df['center_x'] - car_info['center_x']) < 20) & ((df['center_y'] - car_info['center_y']) < 20)
        
        df = df[condition_2]
        if len(df) == 0:
            return car_info['bdbox']
        prev_box = df['bdbox'].values
        prev_box = np.asarray(prev_box.tolist()).mean(axis=0)
        cur_box = np.asarray(car_info['bdbox'])
        new_box = 0.8 * prev_box + 0.2 * cur_box
  
        return new_box.astype(np.int16)
    def save_tracking_info(self):
        fname = '../data/tracking.csv'
        self.df.to_csv(fname)
        print('tracking info saved to {}'.format(fname))
        return
    def __is_car(self, car_info,heat_map):

        if len(self.df)==0:
            car_info['heat_value'] = 0
            return False
        heat_value = heat_map[car_info['center_y'], car_info['center_x']][0]
        car_info['heat_value'] = heat_value
        if  heat_value <= 120:
            return False
        return True
    def __draw_heat_map(self,frame_num):
        
        heat_map_img = np.zeros((720,1280,3), dtype=np.uint8)
        if len(self.df) == 0: 
            return heat_map_img
        last_frame_num  = frame_num - 10
        hot_map_df = self.df[self.df['frame_num'] >= last_frame_num]
        
        for bdbox in hot_map_df['bdbox']:
            x1,y1,x2,y2 = bdbox
            heat_map_img[y1:y2, x1:x2] += 20
            
        return heat_map_img
    
    
 
        
    def run(self):
        
        return


g_frame_tracking= FrameTracking()
if __name__ == "__main__":   
    obj= FrameTracking()
    obj.run()