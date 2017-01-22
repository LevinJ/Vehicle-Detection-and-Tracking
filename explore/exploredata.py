import numpy as np
import os
from os import walk
import pandas as pd




class ExploreData(object):
    def __init__(self):
        self.img_file_csv = '../data/label.csv'
        return
    def __get_all_files_infolder(self, folders):
        f = []
        for folder in folders:
            for (dirpath, _, filenames) in walk(folder):
           
                filenames = [dirpath+ '/' + x for x in filenames if x.endswith('.png')]
                f.extend(filenames)       
        return f
  
    def generate_label_csv(self):
        folders = [r'/home/levin/workspace/carnd/vehichle_detection_tracking/data/dataset/non-vehicles',
                   r'/home/levin/workspace/carnd/vehichle_detection_tracking/data/dataset/vehicles']
        img_files = self.__get_all_files_infolder(folders)
        df = pd.DataFrame(img_files, columns=['FileName'])
        df['label'] = ~df['FileName'].str.contains('non-vehicles')

        df.to_csv(self.img_file_csv)
        print("saved to {}".format(self.img_file_csv))
        return
    def run(self):
        self.generate_label_csv()

        return



if __name__ == "__main__":   
    obj= ExploreData()
    obj.run()