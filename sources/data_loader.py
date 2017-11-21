from flask import Flask, render_template, url_for, request
import json
import time
import data_utils
import os, glob, pdb
import pandas
from SVMRank import SVMRank
#from sklearn.manifold import TSNE
#from sklearn.decomposition import PCA


def main():
    #This sets up the descriptions for each feature. It is a class defined in data_utils
    descriptions = data_utils.load_descriptions()
    #This gets all the standardized data into a dataframe. There is a 'fips_code' column, and each data feature has a corresponding column.
    #You can find the feature names in Data sources/data_values.csv
    #To use a specific column, you use pandas way of calling the feature, i.e. data['CRM250207D']
    data = data_utils.load_data()
    #pca = PCA(n_components=2)

    #Your code here to experiment, test, etc
    print 'Got datastream'
    start = time.time()
    #This gets a subset of features from the dataframe. SO the user provides a set of features (i'll handle how that gets done)
    #For your work, just add or replace elements un the subset_features list below
    #this will get you a dataframe that has columns: 'fips_code', 'AGE040205D','BNK010205D', etc
    query_feature = ['EAN300205D']
    subset_features = query_feature + ['AGE040205D','BNK010205D','BPS030205D','CRM250207D']
    #i.e. subset_features = ['EAN300205D', 'AGE040205D','BNK010205D','BPS030205D','CRM250207D']
    #THis is the function that actually gets you the frame with the features you are focusing on, therefor focus_frame
    focus_frame = data_utils.get_features(subset_features,data)
    
    #sorted_frame = focus_frame.sort_values('EAN300205D', ascending=False)
    #print sorted_frame
    svm = SVMRank(focus_frame, 10)
    weights = svm.train_top_k(False)
    frame_ranks = svm.get_full_rank()
    score = svm.score_test()
    print frame_ranks

    #-----------------------------
    #AND here you work on the frame with your ranking and testing, etc
    #Note that the first entry in subset_features is your query, sp you have to work  on the other frames. 
    #    Maybe extract the frame you need from this. Also, you need to sort query_feature columnin focus frame for testing
    #    Durig sorting you have to keep track of how other columns change - i.e. they should follow query-festure sorting




    #--------------------------------
    print 'Finished with datastream in '+ str(time.time()-start)
    #This is for debugging. 
    pdb.set_trace()


if __name__=='__main__':
    main()
