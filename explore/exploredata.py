import numpy as np
import os
from os import walk
import pandas as pd




class ExploreData(object):
    def __init__(self):
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
        csv_file = '../data/label.csv'
        df.to_csv(csv_file)
        print("saved to {}".format(csv_file))
        return
    def run(self):
        self.generate_label_csv()

        return



if __name__ == "__main__":   
    obj= ExploreData()
    obj.run()