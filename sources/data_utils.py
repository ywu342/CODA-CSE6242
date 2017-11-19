from __future__ import print_function
import sqlite3
#import sklearn
from csv import reader
import pandas as pd
import pdb
import os, glob
import time
SOURCE_FOLDER = './Data sources/'
DESCRIPTION_FILE = os.path.join(SOURCE_FOLDER, 'data_values.csv')
DATA_FOLDER = os.path.join(SOURCE_FOLDER, 'data/')
DATABASE_FILE = './county.db'
class Feature:
    def __init__(self,feat_str):
        self.feat_str = feat_str
        self.feature_id = self.feat_str[0]
        self.feature_desc = self.feat_str[1]
        self.feature_type = self.feat_str[2]
        self.feature_isdecimal = bool(int(self.feat_str[3]))
        self.feature_total = float(self.feat_str[4]) if self.isdecimal() else int(self.feat_str[4])
        self.feature_source = self.feat_str[5]
        self.prefix = self.feature_id[:3]
    def isdecimal(self):
        return self.feature_isdecimal
    def get_prefix(self):
        return self.prefix
    def get_id(self):
        return self.feature_id
    def copy(self):
        return Feature(self.feat_str)


def get_features(subset_features, data):
    return data[['fips_code']+subset_features]
def get_decomposition(subset_features,data,pca):
    #subset_features = list(set([34,5,6,7,8,9,53,56,77,89,43,81,156,176,188,194,143,298,3,32,98]))
    #subset_features = [descriptions.keys()[item] for item in subset_features]
    focus_frame = get_features(subset_features,data)
    focus_result = pca.fit_transform(focus_frame[subset_features].values)
    focus_frame['one_d'] = focus_result[:,0]
    focus_frame['two_d'] = focus_result[:,1]
    focus_frame = focus_frame[['fips_code','one_d','two_d']].set_index('fips_code').T.to_dict('list')
    return focus_frame

def load_data(database_file=DATABASE_FILE, data_folder=DATA_FOLDER):
    file_list = glob.glob(os.path.join(data_folder,'*'))
    file_list = [os.path.splitext(os.path.basename(item))[0] for item in file_list]
    conn = sqlite3.connect(database_file)
    query = 'select fips_code from '+file_list[0]
    start_df = pd.read_sql(query,conn)
    counter = 1
    for table_name in file_list:
        start = time.time()
        query = 'select fips_code, standardized from ' + table_name + ';'
        next_df = pd.read_sql(query,conn)
        start_df = start_df.merge(next_df, on='fips_code',how='inner')
        start_df.rename(index=str,columns={'standardized':table_name}, inplace=True)
        print('Processed '+table_name+'  : '+str(counter)+'/'+str(len(file_list))+' complete in ' +str(time.time()-start), end='\r')
        counter+=1
        #if counter==5:
        #    break
    print ('Done',end='\n')
    return start_df




def load_descriptions(description_file = DESCRIPTION_FILE):
    feature_dict={}
    with open(description_file,'r') as description_:
        description_.readline()
        for line in reader(description_):
            line = Feature(line)
            feature_dict[line.get_id()] = line.copy()
    return feature_dict

def tsne(database_driver, attr_list,county_list):
    return 0