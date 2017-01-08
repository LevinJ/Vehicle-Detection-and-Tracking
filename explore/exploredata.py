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
    def __getlabels(self, file_name):
#         searchObj = re.search(r'(.*)\(.*)_IS(.*)\(.*)\(.*)', file_name, re.M|re.I)
#         searchObj = re.search(r'(.*)_IS(.*)', file_name, re.M|re.I)
        dir_names = file_name.split(self.dir_separator)
        denomation = None
        orientation = dir_names[-1]
        for dir_name in dir_names:
            if '_IS' in dir_name:
                denomation = dir_name
        if denomation is None:
            raise "wrong denomination"
        if not orientation in ['FU', 'FD', 'BU', 'BD']:
            raise "wrong orientation"
        
        
        return denomation, orientation
    def generate_label_csv(self):
        folders = [r'/home/levin/workspace/carnd/vehichle_detection_tracking/data/dataset/non-vehicles/non-vehicles',
                   r'/home/levin/workspace/carnd/vehichle_detection_tracking/data/dataset/vehicles/vehicles']
        img_files = self.__get_all_files_infolder(folders)
        df = pd.DataFrame(img_files, columns=['FileName'])
        df['label'] = ~df['FileName'].str.contains('non-vehicles')
        df.to_csv('../data/label.csv')
        return
    def run(self):
        self.generate_label_csv()

        return



if __name__ == "__main__":   
    obj= ExploreData()
    obj.run()